import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from algorithms.sac import Actor, Critic, gaussian_logprob, weight_init
from HiFNO import HierarchicalFNO, PositionalEncoding, MultiScaleConv, PatchExtractor, Mlp, ConvResFourierLayer, SelfAttention, CrossAttentionBlock, TimeAggregator

class HiFNOEncoder(nn.Module):
    """将HiFNO封装为编码器"""
    def __init__(self, obs_shape, feature_dim, args):
        super().__init__()
        
        self.hifno = HierarchicalFNO(
            img_size=(obs_shape[1], obs_shape[2]),
            patch_size=4,
            in_channels=obs_shape[0],
            out_channels=feature_dim,
            embed_dim=args.embed_dim,
            depth=args.depth if hasattr(args, 'depth') else 2,
            num_scales=args.num_scales,
            truncation_sizes=[16, 12, 8],
            num_heads=4,
            mlp_ratio=2.0,
            activation='gelu'
        )
        
        # 添加时间聚合器处理frame stack
        self.time_aggregator = TimeAggregator(
            embed_dim=feature_dim,
            depth=2,
            num_heads=4,
            num_latents=args.frame_stack,
            mlp_ratio=2.0
        )
        
        # 最终的特征映射层
        self.feature_map = nn.Sequential(
            nn.Linear(feature_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        # 输入形状: (batch_size, channels, height, width)
        features = self.hifno(obs)
        # 时间聚合
        features = self.time_aggregator(features)
        # 映射到所需的特征维度
        features = self.feature_map(features)
        return features

class HiFNOAgent:
    def __init__(self, obs_shape, action_shape, args):
        self.args = args
        self.device = torch.device('cuda')
        
        # 使用封装后的HiFNO编码器
        self.encoder = HiFNOEncoder(
            obs_shape=obs_shape,
            feature_dim=args.embed_dim,
            args=args
        ).to(self.device)

        # 创建Actor和Critic网络
        self.actor = Actor(
            args.hidden_dim,  # 使用映射后的特征维度
            action_shape, 
            args.hidden_dim,
            args.hidden_dim
        ).to(self.device)

        self.critic = Critic(
            args.hidden_dim,
            action_shape,
            args.hidden_dim, 
            args.hidden_dim
        ).to(self.device)

        self.critic_target = Critic(
            args.hidden_dim,
            action_shape,
            args.hidden_dim,
            args.hidden_dim
        ).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 禁用目标网络的梯度
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # 设置优化器
        self.encoder_opt = torch.optim.Adam(
            self.encoder.parameters(), lr=args.lr
        )
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=args.lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.lr
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            features = self.encoder(obs)
            mu = self.actor(features)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            features = self.encoder(obs)
            mu, pi, _, _ = self.actor(features, compute_pi=True)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            features = self.encoder(next_obs)
            next_action = self.actor(features)
            target_Q = self.critic_target(features, next_action)
            target_Q = reward + (not_done * self.args.discount * target_Q)

        features = self.encoder(obs)
        current_Q = self.critic(features, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return critic_loss.item()

    def update_actor(self, obs):
        features = self.encoder(obs)
        action = self.actor(features)
        actor_Q = self.critic(features, action)
        actor_loss = -actor_Q.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return actor_loss.item()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        # 更新critic
        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done)

        # 更新actor
        actor_loss = self.update_actor(obs)

        # 更新目标网络
        utils.soft_update_params(
            self.critic, self.critic_target, self.args.critic_target_tau
        )

        if L is not None:
            L.log('train/critic_loss', critic_loss, step)
            L.log('train/actor_loss', actor_loss, step) 