import torch
import torch.nn.functional as F


from src.utils import schedule
from src.agents.base_agent import BaseAgent


class AWACAgent(BaseAgent):
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
        # awac
        awac_lambda=0.3,
        exp_adv_max=100.0,
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
            dropblock=False,
        )

        self.awac_lambda = awac_lambda
        self.exp_adv_max = exp_adv_max
        self.train()

    def update_actor(self, obs, step, behavioural_action=None):
        metrics = dict()

        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        pi_action = dist.sample(clip=self.stddev_clip)

        with torch.no_grad():
            Q1, Q2 = self.critic(obs, behavioural_action)
            q = torch.min(Q1, Q2)

            Q1, Q2 = self.critic(obs, pi_action)
            v = torch.min(Q1, Q2)

            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self.awac_lambda), self.exp_adv_max
            )

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

        metrics["actor_logprob"] = log_prob.mean().item()
        metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
        metrics["actor_bc_loss"] = actor_bc_loss.item()
        metrics["actor_loss"] = actor_loss.item()

        return metrics
