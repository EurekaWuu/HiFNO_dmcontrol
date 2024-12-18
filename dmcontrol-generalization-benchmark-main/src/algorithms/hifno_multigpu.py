import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import algorithms.modules as m
from algorithms.sac import SAC

from algorithms.modules import Actor, Critic
from algorithms.models.HiFNO import HierarchicalFNO, PositionalEncoding, MultiScaleConv, PatchExtractor, Mlp, ConvResFourierLayer, SelfAttention, CrossAttentionBlock, TimeAggregator

import os
import time
from datetime import datetime

class HiFNOEncoder(nn.Module):
    """将HiFNO封装为编码器"""
    def __init__(self, obs_shape, feature_dim, args):
        super().__init__()
        
        self.out_dim = args.hidden_dim
        self.device = torch.device('cuda')
        
        # 获取每帧的通道数
        self.frame_stack = args.frame_stack
        self.channels_per_frame = obs_shape[0] // self.frame_stack  # 每帧的通道数(RGB=3)
        
        # 创建HiFNO - 检查输入通道数
        self.hifno = HierarchicalFNO(
            img_size=(obs_shape[1], obs_shape[2]),
            patch_size=4,
            in_channels=self.frame_stack * 3,  # 使用frame_stack * 3作为输入通道数
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
        
        # 特征映射层
        self.feature_map = nn.Sequential(
            nn.Linear(args.embed_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU()
        )
        
        self.to(self.device)

    def to(self, device):
        super().to(device)
        self.hifno.to(device)
        self.time_aggregator.to(device)
        self.feature_map.to(device)
        return self

    def forward(self, obs, detach=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        
        # 处理输入维度
        if len(obs.shape) == 3:  # (C, H, W)
            obs = obs.unsqueeze(0)  # 添加batch维度 -> (B, C, H, W)
        elif len(obs.shape) == 2:  # (H, W)
            obs = obs.unsqueeze(0).unsqueeze(0)  # -> (B, 1, H, W)
            obs = obs.repeat(1, self.frame_stack * 3, 1, 1)  # 扩展到正确的通道数
        
        obs = obs.to(self.device)
        
        # 检查输入维度
        B, C, H, W = obs.shape
        expected_channels = self.frame_stack * 3
        if C != expected_channels:
            # 如果通道数不正确，调整到正确的通道数
            if C == 1:
                obs = obs.repeat(1, expected_channels, 1, 1)
            elif C == 3:
                obs = obs.repeat(1, self.frame_stack, 1, 1)
            else:
                raise ValueError(f"Unexpected number of channels: {C}, expected {expected_channels}")
        
        # 通过HiFNO处理
        features = self.hifno(obs)  # (B, feature_dim, H', W')
        
        # 重塑为时序数据
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # (B, H, W, feature_dim)
        
        # 时间聚合
        features = features.unsqueeze(1)  # (B, 1, H, W, feature_dim)
        features = self.time_aggregator(features)  # (B, T', H, W, embed_dim)
        
        # 空间平均池化
        features = features.mean(dim=(1, 2, 3))  # (B, embed_dim)
        
        # 特征映射
        features = self.feature_map(features)  # (B, hidden_dim)
        
        if detach:
            features = features.detach()
        
        return features

    def copy_conv_weights_from(self, source):
        """
        从另一个编码器复制卷积权重
        """
        pass

class HiFNOAgent(SAC):
    def __init__(self, obs_shape, action_shape, args):
        self.args = args
        self.device = torch.device('cuda')
        
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if hasattr(args, 'work_dir'):
            args.work_dir = os.path.join(args.work_dir, f'{timestamp}')
            if not os.path.exists(args.work_dir):
                os.makedirs(args.work_dir)
        
        
        self.encoder = HiFNOEncoder(
            obs_shape=obs_shape,
            feature_dim=args.embed_dim,
            args=args
        ).to(self.device)

        # 保存动作维度
        self.action_shape = action_shape
        
        
        self.actor = m.Actor(
            self.encoder,
            action_shape, 
            args.hidden_dim,
            args.actor_log_std_min,
            args.actor_log_std_max
        ).to(self.device)

        self.critic = m.Critic(
            self.encoder,
            action_shape,
            args.hidden_dim
        ).to(self.device)

        self.critic_target = m.Critic(
            self.encoder,
            action_shape,
            args.hidden_dim
        ).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 从SAC中复制必要的初始化
        self.log_alpha = torch.tensor(np.log(args.init_temperature)).to(self.device)
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
            if isinstance(self.actor, torch.nn.DataParallel):
                # 获取编码后的特征
                next_obs_encoded = self.actor.module.encoder(next_obs)
                _, policy_action, log_pi, _ = self.actor.module(next_obs)
                target_Q1, target_Q2 = self.critic_target.module(next_obs_encoded, policy_action)
            else:
                next_obs_encoded = self.actor.encoder(next_obs)
                _, policy_action, log_pi, _ = self.actor(next_obs)
                target_Q1, target_Q2 = self.critic_target(next_obs_encoded, policy_action)
            
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.args.discount * target_V)

        if isinstance(self.critic, torch.nn.DataParallel):
            obs_encoded = self.critic.module.encoder(obs)
            current_Q1, current_Q2 = self.critic.module(obs_encoded, action)
        else:
            obs_encoded = self.critic.encoder(obs)
            current_Q1, current_Q2 = self.critic(obs_encoded, action)
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if L is not None:
            L.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

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
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.reshape(1, *obs.shape)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.reshape(1, 1, *obs.shape)  # 添加batch和channel维度
            
            obs = torch.FloatTensor(obs).to(self.device)
            
            if isinstance(self.actor, torch.nn.DataParallel):
                # 直接使用原始观察，让Actor内部处理
                mu, _, _, _ = self.actor.module(obs, compute_pi=False, compute_log_pi=False)
            else:
                mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            
            action = mu.cpu().data.numpy().flatten()
            action = np.clip(action, -1.0, 1.0)
            action = action[:self.action_shape[0]]
            return action

    def sample_action(self, obs):
        with torch.no_grad():
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.reshape(1, *obs.shape)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.reshape(1, 1, *obs.shape)  # 添加batch和channel维度
            
            obs = torch.FloatTensor(obs).to(self.device)
            
            if isinstance(self.actor, torch.nn.DataParallel):
                # 直接使用原始观察，让Actor内部处理
                _, pi, _, _ = self.actor.module(obs, compute_log_pi=False)
            else:
                _, pi, _, _ = self.actor(obs, compute_log_pi=False)
            
            action = pi.cpu().data.numpy().flatten()
            action = np.clip(action, -1.0, 1.0)
            action = action[:self.action_shape[0]]
            return action

    def update_actor_and_alpha(self, obs, L=None, step=None):
        if isinstance(self.actor, torch.nn.DataParallel):
            obs_encoded = self.actor.module.encoder(obs)
            _, pi, log_pi, log_std = self.actor.module(obs)
            actor_Q1, actor_Q2 = self.critic.module(obs_encoded, pi)
        else:
            obs_encoded = self.actor.encoder(obs)
            _, pi, log_pi, log_std = self.actor(obs)
            actor_Q1, actor_Q2 = self.critic(obs_encoded, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
            L.log('train_actor/entropy', -log_pi.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                     (-log_pi - self.target_entropy).detach()).mean()
        if L is not None:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)

        alpha_loss.backward()
        self.log_alpha_optimizer.step()