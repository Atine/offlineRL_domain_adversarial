import torch
import torch.nn.functional as F


from src.utils import soft_update_params
from src.agents.base_agent import BaseAgent
from src.models import Encoder, ClassifierWithLogitsRev


class DrQv2TogAgent(BaseAgent):
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

        # models
        self.encoder = Encoder(obs_dim, dropblock=dropblock).to(
            device
        )
        self.classifier = ClassifierWithLogitsRev(
            self.encoder.repr_dim, 1, feature_dim, hidden_dim
        ).to(device)

        # optimisers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.class_opt = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        # others
        self.criterion = torch.nn.BCELoss()
        self.models_list = ["encoder", "actor", "critic", "classifier"]

        self.train()

    def train(self, training=True):
        super().train(training)
        self.classifier.train(training)

    def update_critic(
        self,
        enc_obs_rev,
        enc_obs,
        enc_obs_dis_rev,
        enc_obs_dis,
        enc_next_obs_dis,
        action_dis,
        reward_dis,
        discount_dis,
        step,
        label=1.0,
        label_dis=0.0,
        flooding=0.0,
    ):
        metrics = dict()
        b_size = enc_obs_rev.shape[0]
        label = torch.ones((b_size, 1)).float().to(self.device) * label
        label_dis = torch.ones((b_size, 1)).float().to(self.device) * label_dis

        # real labels
        class_obs, logits = self.classifier(enc_obs_rev)
        D_real_loss = self.criterion(class_obs, label)

        # fake labels
        class_obs_dis, logits_dis = self.classifier(enc_obs_dis_rev)
        D_fake_loss = self.criterion(class_obs_dis, label_dis)

        # all classifier loss
        class_loss = D_real_loss + D_fake_loss
        class_loss = (class_loss - flooding).abs() + flooding

        with torch.no_grad():
            dist = self.actor(enc_next_obs_dis)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(
                enc_next_obs_dis, next_action
            )
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward_dis.float() + (discount_dis * target_V)

        Q1, Q2 = self.critic(enc_obs_dis, action_dis)
        qf1_loss = F.mse_loss(Q1, target_Q)
        qf2_loss = F.mse_loss(Q2, target_Q)

        # all critic loss
        critic_loss = qf1_loss + qf2_loss

        # all loss
        all_loss = class_loss + critic_loss

        # log metrics
        metrics["L1"] = class_obs.mean().item()
        metrics["L2"] = class_obs_dis.mean().item()
        metrics["critic_target_q"] = target_Q.mean().item()
        metrics["critic_q1"] = Q1.mean().item()
        metrics["critic_q2"] = Q2.mean().item()
        metrics["critic_loss"] = critic_loss.item()
        metrics["class_loss"] = class_loss.item()

        # optimise encoder and critic
        self.encoder_opt.zero_grad()
        self.class_opt.zero_grad()
        self.critic_opt.zero_grad()
        all_loss.backward()
        self.critic_opt.step()
        self.class_opt.step()
        self.encoder_opt.step()

        return metrics

    def update(self, replay_buffer, replay_buffer_dis, step, batch_size):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        obs, _, reward, _, _ = self.sample(replay_buffer, batch_size)
        (
            obs_dis,
            action_dis,
            reward_dis,
            discount_dis,
            next_obs_dis,
        ) = self.sample(replay_buffer_dis, batch_size)
        metrics["batch_reward"] = reward.mean().item()
        metrics["batch_reward_dis"] = reward_dis.mean().item()

        # augment
        obs, shift = self.aug(obs.float(), shift=None)
        obs_dis, _ = self.aug(obs_dis.float(), shift=shift)
        next_obs_dis, _ = self.aug(next_obs_dis.float(), shift=shift)

        # encode
        enc_obs_rev, enc_obs = self.encoder(obs, scale=-1.0)
        enc_obs_dis_rev, enc_obs_dis = self.encoder(obs_dis, scale=-1.0)
        with torch.no_grad():
            _, enc_next_obs_dis = self.encoder(next_obs_dis)

        # update critic
        metrics.update(
            self.update_critic(
                enc_obs_rev,
                enc_obs,
                enc_obs_dis_rev,
                enc_obs_dis,
                enc_next_obs_dis,
                action_dis,
                reward_dis,
                discount_dis,
                step,
            )
        )

        # update actor
        metrics.update(
            self.update_actor(enc_obs_dis.detach(), step, action_dis.detach())
        )

        # update critic target
        soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
