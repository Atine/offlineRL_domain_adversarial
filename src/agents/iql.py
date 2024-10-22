import torch
import torch.nn.functional as F


from src.utils import schedule, soft_update_params
from src.agents.base_agent import BaseAgent
from src.models.values import Value


class IQLAgent(BaseAgent):
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_bc=True,
        bc_weight=2.5,
        # ours
        augmentation=1,
        dropblock=0.0,
        # iql
        iql_scale=3.0,
        iql_expectile=0.7,
        schedule_T_max=1000000,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            device,
            lr,
            feature_dim,
            hidden_dim,
            critic_target_tau,
            update_every_steps,
            stddev_schedule,
            stddev_clip,
            use_bc,
            bc_weight,
            augmentation,
            dropblock=dropblock,
        )

        self.value = Value(self.encoder.repr_dim).to(device)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_opt, T_max=int(schedule_T_max)
        )
        self.iql_scale = iql_scale
        self.iql_expectile = iql_expectile
        self.models_list = ["encoder", "actor", "critic", "value"]
        self.train()

    def train(self, training=True):
        super().train(training)
        self.value.train(training)

    def update_value(self, obs, action):
        metrics = dict()

        # encode
        _, encoded_obs = self.encoder(obs)

        with torch.no_grad():
            q1, q2 = self.critic_target(encoded_obs, action)
            q = torch.minimum(q1, q2).detach()

        v = self.value(encoded_obs)
        value_loss = self.expectile_loss(q - v, self.iql_expectile).mean()

        # optimise encoder and value
        self.encoder_opt.zero_grad()
        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()
        self.encoder_opt.step()

        metrics["value_loss"] = value_loss.item()
        return metrics

    def update_critic(
        self, obs, action, reward, discount, next_obs, step, noise=None
    ):
        metrics = dict()

        # encode
        _, encoded_obs = self.encoder(obs)
        _, encoded_next_obs = self.encoder(next_obs)
        encoded_next_obs = encoded_next_obs.detach()

        with torch.no_grad():
            next_v = self.value(encoded_next_obs)
            target_Q = reward.float() + (discount * next_v)

        Q1, Q2 = self.critic(encoded_obs, action)
        qf1_loss = F.mse_loss(Q1, target_Q)
        qf2_loss = F.mse_loss(Q2, target_Q)

        # all critic loss
        critic_loss = qf1_loss + qf2_loss

        # log metrics
        metrics["critic_target_q"] = target_Q.mean().item()
        metrics["critic_q1"] = Q1.mean().item()
        metrics["critic_q2"] = Q2.mean().item()
        metrics["critic_loss"] = critic_loss.item()

        # optimise encoder and critic
        self.encoder_opt.zero_grad()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, behavioural_action=None):
        metrics = dict()

        # encode
        _, encoded_obs = self.encoder(obs)

        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(encoded_obs, stddev)
        pi_action = dist.sample(clip=self.stddev_clip)

        with torch.no_grad():
            Q1, Q2 = self.critic_target(encoded_obs, behavioural_action)
            q = torch.min(Q1, Q2)
            v = self.value(encoded_obs)
            adv = q - v
            weights = torch.exp(self.iql_scale * (adv)).clamp(0, 100)

        log_prob = dist.log_prob(behavioural_action).sum(-1, keepdim=True)
        actor_loss = (-log_prob * weights).mean()

        # offline BC Loss
        actor_bc_loss = F.mse_loss(pi_action, behavioural_action)
        # Eq. 5 of arXiv:2106.06860
        if self.use_bc:
            lam = self.bc_weight / v.detach().abs().mean()
            actor_loss = actor_loss * lam + actor_bc_loss
        else:
            actor_loss = actor_loss

        # optimise actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.actor_scheduler.step()

        metrics["actor_logprob"] = log_prob.mean().item()
        metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
        metrics["actor_bc_loss"] = actor_bc_loss.item()
        metrics["actor_loss"] = actor_loss.item()

        return metrics

    def update(self, replay_buffer, step, batch_size):
        """Need to reimplement update() for IQL because we need to calculate
        twice the encoded features from encoder(obs) for updating in
        value network and critic network
        """

        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, discount, next_obs = self.sample(
            replay_buffer, batch_size
        )
        metrics["batch_reward"] = reward.mean().item()

        # augment
        obs, shift = self.aug(obs.float(), shift=None)
        next_obs, _ = self.aug(next_obs.float(), shift=shift)

        # update value
        metrics.update(self.update_value(obs, action))

        # update critic
        metrics.update(
            self.update_critic(
                obs, action, reward, discount, next_obs, step, noise=None
            )
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step, action.detach()))

        # update critic target
        soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def expectile_loss(self, diff, expectile):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)
