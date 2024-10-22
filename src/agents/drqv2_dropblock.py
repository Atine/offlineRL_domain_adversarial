import torch
import torch.nn.functional as F


from src.utils import schedule
from src.agents.base_agent import BaseAgent
from src.models import ActorDefault


class DrQv2DropblockAgent(BaseAgent):
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
        dropblock=0.3,
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

        self.actor = ActorDefault(
            self.encoder.repr_dim, action_dim, feature_dim, hidden_dim
        ).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def act_with_logprob(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).float()
        obs, _ = self.encoder(obs.unsqueeze(0))
        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)

        if eval_mode:
            action = dist.mean
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        else:
            x_t = dist.rsample()
            actions = torch.tanh(x_t)
            log_prob = dist.log_prob(x_t)
            log_prob -= torch.log((1 - actions.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

        return action.cpu().numpy()[0], log_prob

    def update_critic(
        self, enc_obs, action, reward, discount, enc_next_obs, step
    ):
        metrics = dict()

        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(enc_next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(
                enc_next_obs, next_action
            )
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward.float() + (discount * target_V)

        Q1, Q2 = self.critic(enc_obs, action)
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

        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # offline BC Loss
        actor_bc_loss = F.mse_loss(action, behavioural_action)
        # Eq. 5 of arXiv:2106.06860
        if self.use_bc:
            lam = self.bc_weight / Q.detach().abs().mean()
            actor_loss = actor_loss * lam + actor_bc_loss
        else:
            actor_loss = actor_loss

        # log metrics
        metrics["actor_logprob"] = log_prob.mean().item()
        metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
        metrics["actor_bc_loss"] = actor_bc_loss.item()
        metrics["actor_loss"] = actor_loss.item()

        # optimise actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return metrics
