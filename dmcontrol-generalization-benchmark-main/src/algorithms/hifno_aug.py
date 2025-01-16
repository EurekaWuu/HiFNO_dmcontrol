import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import algorithms.modules_multigpu as m
from algorithms.sac_multigpu import SAC
import augmentations

from algorithms.modules_multigpu import Actor, Critic
from algorithms.models.HiFNO_multigpu import HierarchicalFNO, PositionalEncoding, MultiScaleConv, PatchExtractor, Mlp, ConvResFourierLayer, SelfAttention, CrossAttentionBlock, TimeAggregator

import os
import time
from datetime import datetime
import torch.distributed as dist

class HiFNOEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, args):
        super().__init__()
        self.args = args
        
        self.out_dim = args.hidden_dim
        self.frame_stack = args.frame_stack
        self.channels_per_frame = 3
        
        
        self.hifno = HierarchicalFNO(
            img_size=(obs_shape[1], obs_shape[2]),
            patch_size=4,
            in_channels=self.frame_stack * self.channels_per_frame,
            out_channels=feature_dim,
            embed_dim=args.embed_dim,
            depth=args.depth if hasattr(args, 'depth') else 2,
            num_scales=args.num_scales if hasattr(args, 'num_scales') else 3, 
            truncation_sizes=[16, 12, 8],
            num_heads=4,
            mlp_ratio=2.0,
            activation='gelu'
        )
        
        # 时间聚合
        self.time_aggregator = TimeAggregator(
            embed_dim=args.embed_dim,
            depth=2,
            num_heads=4,
            num_latents=args.frame_stack,
            mlp_ratio=2.0
        )
        
        # 特征映射，确保输出维度是 hidden_dim
        self.feature_map = nn.Sequential(
            nn.Linear(args.embed_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU()
        )

    def forward(self, obs, detach=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        
        if len(obs.shape) == 2:
            return obs
        
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        
        # 如果是五维 (B, T, C, H, W) 且 T=1，可直接 squeeze(1) 变成 (B, C, H, W)
        if obs.dim() == 5 and obs.shape[1] == 1:
            obs = obs.squeeze(1)
        
        B, C, H, W = obs.shape
        expected_channels = self.frame_stack * self.channels_per_frame
        
        if C != expected_channels:
            if C == self.channels_per_frame:
                obs = obs.repeat(1, self.frame_stack, 1, 1)
            else:
                raise ValueError(f"Unexpected number of input channels: {C}, expected {expected_channels}")
        
        
        device = next(self.parameters()).device
        obs = obs.to(device)
        
        # 特征提取
        features = self.hifno(obs)
        features = features.permute(0, 2, 3, 1)
        features = features.unsqueeze(1)
        features = self.time_aggregator(features)
        features = features.mean(dim=(1, 2, 3, 4))
        features = self.feature_map(features)
        
        if detach:
            features = features.detach()
        
        return features

    def copy_conv_weights_from(self, source):
        """
        从另一个编码器复制卷积权重
        """
        pass

class HiFNOAgent(SAC, nn.Module):
    def __init__(self, obs_shape, action_shape, args):
        nn.Module.__init__(self)
        
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        
        self.args = args
        self.action_shape = action_shape
        
        
        self.encoder = HiFNOEncoder(
            obs_shape=obs_shape,
            feature_dim=args.embed_dim,
            args=args
        )
        
        self.actor = m.Actor(
            self.encoder,
            action_shape, 
            args.hidden_dim,
            args.actor_log_std_min,
            args.actor_log_std_max
        )

        self.critic = m.Critic(
            self.encoder,
            action_shape,
            args.hidden_dim
        )

        self.critic_target = m.Critic(
            self.encoder,
            action_shape,
            args.hidden_dim
        )
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        
        self.log_alpha = torch.tensor(np.log(args.init_temperature))
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)
        
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.args.discount * target_V)

        action = action.view(action.shape[0], -1)

        if self.args.alpha == self.args.beta:
            obs_aug = augmentations.random_conv(obs.clone())
            obs_cat = utils.cat(obs, obs_aug)
            action_cat = utils.cat(action, action)
            target_Q_cat = utils.cat(target_Q, target_Q)

            current_Q1, current_Q2 = self.critic(obs_cat, action_cat)
            critic_loss = (self.args.alpha + self.args.beta) * (
                F.mse_loss(current_Q1, target_Q_cat) + F.mse_loss(current_Q2, target_Q_cat)
            )
        else:
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = self.args.alpha * (
                F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            )

            obs_aug = augmentations.random_conv(obs.clone())
            current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
            critic_loss += self.args.beta * (
                F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q)
            )

        if L is not None:
            L.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        
        # # 打印原始形状
        # print("Original shapes:")
        # print(f"obs: {obs.shape}")
        # print(f"next_obs: {next_obs.shape}")

        # 将numpy数组转换为tensor，但不指定设备
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.FloatTensor(next_obs)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
        if isinstance(reward, np.ndarray):
            reward = torch.FloatTensor(reward)
        if isinstance(not_done, np.ndarray):
            not_done = torch.FloatTensor(not_done)

        
        #   如果 obs 已经是 (B,9,H,W)，则不需要 permute
        #   如果 obs.shape = (B,H,W,9)，才做 permute
        if obs.dim() == 4:
            if obs.shape[1] != 9 and obs.shape[3] == 9:  # (B,H,W,9) => (B,9,H,W)
                obs = obs.permute(0, 3, 1, 2)

            if next_obs.shape[1] != 9 and next_obs.shape[3] == 9:
                next_obs = next_obs.permute(0, 3, 1, 2)

        
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.args.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.args.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.args.critic_tau
            )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.FloatTensor(obs)

            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
            
            
            device = next(self.parameters()).device
            obs = obs.to(device)

            
            mu, _, _, _ = self.actor(obs)
            
            action = mu.cpu().data.numpy().flatten()
            action = np.clip(action, -1.0, 1.0)
            action = action[:self.action_shape[0]]
            return action

    def sample_action(self, obs):
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.FloatTensor(obs)
            
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
            elif len(obs.shape) == 2:
                obs = obs.unsqueeze(0).unsqueeze(0)
            
            device = next(self.parameters()).device
            obs = obs.to(device)
            
            # 直接调用 actor
            _, pi, _, _ = self.actor(obs)
            
            action = pi.cpu().data.numpy().flatten()
            action = np.clip(action, -1.0, 1.0)
            action = action[:self.action_shape[0]]
            return action

    def update_actor_and_alpha(self, obs, L=None, step=None):
        # 不再区分 DataParallel
        obs_encoded = self.actor.encoder(obs)
        _, pi, log_pi, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs_encoded, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
            L.log('train_actor/entropy', -log_pi.mean(), step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        
        if L is not None:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)

        alpha_loss.backward()
        self.log_alpha_optimizer.step()