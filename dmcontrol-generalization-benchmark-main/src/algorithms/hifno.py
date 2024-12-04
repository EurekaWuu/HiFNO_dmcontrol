import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import algorithms.modules as m
from algorithms.sac import SAC

from algorithms.my_modules import Actor, Critic
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
        self.channels_per_frame = obs_shape[0] // self.frame_stack
        
        # 创建HiFNO
        self.hifno = HierarchicalFNO(
            img_size=(obs_shape[1], obs_shape[2]),
            patch_size=4,
            in_channels=obs_shape[0],
            out_channels=feature_dim,  # 使用feature_dim作为输出维度
            embed_dim=args.embed_dim,
            depth=args.depth if hasattr(args, 'depth') else 2,
            num_scales=args.num_scales,
            truncation_sizes=[16, 12, 8],
            num_heads=4,
            mlp_ratio=2.0,
            activation='gelu'
        )
        
        # 添加时间聚合器
        self.time_aggregator = TimeAggregator(
            embed_dim=args.embed_dim,  # 目标维度
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
        
        # 将所有组件移动到GPU
        self.to(self.device)

    def to(self, device):
        super().to(device)
        self.hifno.to(device)
        self.time_aggregator.to(device)
        self.feature_map.to(device)
        return self

    def forward(self, obs, detach=False):
        # 确保输入在正确的设备上并且维度正确
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        
        # 处理输入维度
        if len(obs.shape) == 3:  # (C, H, W)
            obs = obs.unsqueeze(0)  # 添加batch维度
        elif len(obs.shape) == 2:  # (H, W)
            obs = obs.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        obs = obs.to(self.device)
        
        # 通过HiFNO处理
        features = self.hifno(obs)  # (B, feature_dim, H', W')
        
        # 重塑为时序数据，保持空间维度的结构
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # (B, H, W, feature_dim)
        
        # 时间聚合（TimeAggregator会自动处理维度调整）
        features = features.unsqueeze(1)  # (B, 1, H, W, feature_dim)
        features = self.time_aggregator(features)  # (B, T', H, W, embed_dim)
        
        # 取平均
        features = features.mean(dim=(1, 2, 3))  # (B, embed_dim)
        
        # 映射到所需的特征维度
        features = self.feature_map(features)  # (B, hidden_dim=6)
        
        # 确保最终输出维度正确
        if len(features.shape) > 2:
            features = features.mean(dim=1)  # 如果还有额外维度，取平均
        
        if detach:
            features = features.detach()
        
        # 打印维度以便调试
        print(f"Encoder output shape: {features.shape}")
            
        return features

    def copy_conv_weights_from(self, source):
        """
        从另一个编码器复制卷积权重
        """
        pass

class HiFNOAgent(SAC):
    def __init__(self, obs_shape, action_shape, args):
        # 不直接调用SAC的__init__，而是重新实现
        self.args = args
        self.device = torch.device('cuda')
        
        # 修改工作目录，添加时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if hasattr(args, 'work_dir'):
            args.work_dir = os.path.join(args.work_dir, f'{timestamp}')
            if not os.path.exists(args.work_dir):
                os.makedirs(args.work_dir)
        
        # 使用封装后的HiFNO编码器
        self.encoder = HiFNOEncoder(
            obs_shape=obs_shape,
            feature_dim=args.embed_dim,
            args=args
        ).to(self.device)

        # 保存动作维度
        self.action_shape = action_shape
        
        # 创建Actor和Critic网络
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
        
        # 设置优化器
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

        current_Q1, current_Q2 = self.critic(obs, action)
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
            # 将LazyFrames转换为numpy数组
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            # 处理输入维度
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.reshape(1, *obs.shape)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.reshape(1, 1, *obs.shape)  # 添加batch和channel维度
            
            obs = torch.FloatTensor(obs).to(self.device)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            # 确保动作维度正确
            action = mu.cpu().data.numpy().flatten()
            # 限制动作范围在[-1, 1]之间
            action = np.clip(action, -1.0, 1.0)
            # 只取前n个维度，其中n是环境动作空间的维度
            action = action[:self.action_shape[0]]
            return action

    def sample_action(self, obs):
        with torch.no_grad():
            # 将LazyFrames转换为numpy数组
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            # 处理输入维度
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.reshape(1, *obs.shape)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.reshape(1, 1, *obs.shape)  # 添加batch和channel维度
            
            obs = torch.FloatTensor(obs).to(self.device)
            _, pi, _, _ = self.actor(obs, compute_log_pi=False)
            # 确保动作维度正确
            action = pi.cpu().data.numpy().flatten()
            # 限制动作范围在[-1, 1]之间
            action = np.clip(action, -1.0, 1.0)
            # 只取前n个维度，其中n是环境动作空间的维度
            action = action[:self.action_shape[0]]
            return action