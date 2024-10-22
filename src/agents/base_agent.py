import os
import torch
import torch.nn.functional as F


from src.utils import to_torch, soft_update_params, augmentator
from src.models import Encoder, ActorSimple, Critic


class BaseAgent(object):
    """BaseAgent class for RL algorithms. Implementation style
       vaguely follows DrQv2

    Default implemented methods:
        train()
        act_with_logprob()
        update_critic()
        update()
        save_models()
        update_actor()  (DrQv2-like actor update)

    update_critic() and update() may also require overwrite if necessary

    """

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
        dropblock=False,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropblock = dropblock

        self.device = torch.device(device)
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.bc_weight = bc_weight
        self.use_bc = use_bc

        # models
        self.encoder = Encoder(obs_dim, dropblock=dropblock).to(device)
        self.actor = ActorSimple(
            self.encoder.repr_dim, action_dim, feature_dim, hidden_dim
        ).to(device)
        self.critic = Critic(
            self.encoder.repr_dim, action_dim, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            self.encoder.repr_dim, action_dim, feature_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.models_list = ["encoder", "actor", "critic"]

        # optimisers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = (
            augmentator[augmentation](obs_dim[0] // 3)
            if augmentation
            else lambda x: x
        )

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)

    def act_with_logprob(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).float()
        obs, _ = self.encoder(obs.unsqueeze(0))
        dist = self.actor(obs)

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

    def get_ave_q_prediction_for_bias(self, obs_tensor, acts_tensor):
        # use min Q for bias evaluation
        obs_tensor, _ = self.encoder(obs_tensor)
        q1, q2 = self.critic(obs_tensor, acts_tensor)
        q_prediction = torch.min(q1, q2)
        average_q_prediction = torch.mean(q_prediction, dim=1)
        return average_q_prediction

    def update_critic(
        self, enc_obs, action, reward, discount, enc_next_obs, step
    ):
        """Simple double-Q critic is implemented
        Expects to use enc_obs and enc_next_obs
        from encoder() to train critic.
        In some cases (such as in IQL) need to reimplement this function
        """

        metrics = dict()

        with torch.no_grad():
            dist = self.actor(enc_next_obs)
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

        dist = self.actor(obs)
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

    def sample(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
        obs, action, reward, discount, next_obs = to_torch(batch, self.device)

        # normalise images to [0, 1]
        obs = obs / 255.0
        next_obs = next_obs / 255.0
        return obs, action, reward, discount, next_obs

    def update(self, replay_buffer, step, batch_size):
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

        # encode
        _, enc_obs = self.encoder(obs)
        with torch.no_grad():
            _, enc_next_obs = self.encoder(next_obs)

        # update critic
        metrics.update(
            self.update_critic(
                enc_obs, action, reward, discount, enc_next_obs, step
            )
        )

        # update actor
        metrics.update(
            self.update_actor(enc_obs.detach(), step, action.detach())
        )

        # update critic target
        soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def calculate_bellman_errors(self, input_replay_buffer, step):
        errors = []
        for _ in range(200):
            obs, action, reward, discount, next_obs = self.sample(
                input_replay_buffer, 256
            )
            _, enc_obs = self.encoder(obs)
            _, enc_next_obs = self.encoder(next_obs)

            with torch.no_grad():
                dist = self.actor(enc_next_obs)
                next_action = dist.mean

            next_Q1, next_Q2 = self.critic(enc_next_obs, next_action)
            Q1, Q2 = self.critic(enc_obs, action)
            bellman_error = (reward + discount * next_Q1 - Q1).detach()
            errors.append(bellman_error)

        errors = torch.cat(errors, 0)
        return errors.cpu().numpy()

    def calculate_norms(self):
        metrics = {}
        for model in self.models_list:
            sum_param_norm = 0.0
            sum_gradient_norm = 0.0
            for param in getattr(self, model).parameters():
                param_norm = param.data.norm(2)
                gradient_norm = param.grad.detach().data.norm(2)
                sum_param_norm += param_norm.item() ** 2
                sum_gradient_norm += gradient_norm.item() ** 2

            sum_param_norm = sum_param_norm**0.5
            sum_gradient_norm = sum_gradient_norm**0.5
            metrics[f"{model}_param_norm"] = sum_param_norm
            metrics[f"{model}_gradient_norm"] = sum_gradient_norm

        return metrics

    def save_models(self, step, work_dir):
        for model in self.models_list:
            os.makedirs(os.path.join(work_dir, model), exist_ok=True)
            getattr(self, model).save(
                os.path.join(work_dir, f"{model}/{step}.pth")
            )
