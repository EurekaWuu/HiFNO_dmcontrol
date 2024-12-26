import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import algorithms.modules as m
import augmentations
from algorithms.sac import SAC
from bisimulation_loss import BisimulationLoss

from algorithms.modules import Actor, Critic
from algorithms.models.HiFNO import HierarchicalFNO, PositionalEncoding, MultiScaleConv, PatchExtractor, Mlp, ConvResFourierLayer, SelfAttention, CrossAttentionBlock, TimeAggregator

import os
import time
from datetime import datetime

class HiFNOEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, args):
        super().__init__()
        
        self.out_dim = args.hidden_dim
        self.device = torch.device('cuda')
        
        # 获取每帧的通道数
        self.frame_stack = args.frame_stack
        self.channels_per_frame = obs_shape[0] // self.frame_stack
        
        
        self.hifno = HierarchicalFNO(
            img_size=(obs_shape[1], obs_shape[2]),
            patch_size=4,
            in_channels=obs_shape[0],
            out_channels=feature_dim,  # feature_dim作为输出维度
            embed_dim=args.embed_dim,
            depth=args.depth if hasattr(args, 'depth') else 2,
            num_scales=args.num_scales,
            truncation_sizes=[16, 12, 8],
            num_heads=4,
            mlp_ratio=2.0,
            activation='gelu'
        )
        
        
        self.time_aggregator = TimeAggregator(
            embed_dim=args.embed_dim,  # 目标维度
            depth=2,
            num_heads=4,
            num_latents=args.frame_stack,
            mlp_ratio=2.0
        )
        
        
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
            obs = obs.unsqueeze(0)  # 添加batch维度
        elif len(obs.shape) == 2:  # (H, W)
            obs = obs.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        obs = obs.to(self.device)
        
        
        features = self.hifno(obs)  # (B, feature_dim, H', W')
        
        # 重塑为时序数据，保持空间维度的结构
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # (B, H, W, feature_dim)
        
        # 时间聚合
        features = features.unsqueeze(1)  # (B, 1, H, W, feature_dim)
        features = self.time_aggregator(features)  # (B, T', H, W, embed_dim)
        
        
        features = features.mean(dim=(1, 2, 3))  # (B, embed_dim)
        
        # 映射到所需的特征维度
        features = self.feature_map(features)  # (B, hidden_dim=6)
        
        # 检查最终输出维度
        if len(features.shape) > 2:
            features = features.mean(dim=1)  # 如果还有额外维度，取平均
        
        if detach:
            features = features.detach()
        
        
        # print(f"Encoder output shape: {features.shape}")
            
        return features

    def copy_conv_weights_from(self, source):
        pass

class HiFNOBisimAgent(SAC):
    """带有双模拟损失的HiFNO代理"""
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

        
        self.action_shape = action_shape
        
        
        self.actor = m.BisimActor(
            self.encoder,
            action_shape, 
            args.hidden_dim,
            args.actor_log_std_min,
            args.actor_log_std_max
        ).to(self.device)

        self.critic = m.BisimCritic(
            self.encoder,
            action_shape,
            args.hidden_dim
        ).to(self.device)

        self.critic_target = m.BisimCritic(
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

        self.bisim_loss_fn = BisimulationLoss(
            lambda_BB=getattr(args, 'lambda_BB', 0.8),
            lambda_ICC=getattr(args, 'lambda_ICC', 0.4),
            lambda_CC=getattr(args, 'lambda_CC', 0.4),
            p=getattr(args, 'bisim_p', 2),
            check_inputs=True
        )

    def compute_bisim_loss(self, obs_batch):
        # 将批次分成两组状态
        batch_size = obs_batch.shape[0] // 2
        obs_i = obs_batch[:batch_size]
        obs_j = obs_batch[batch_size:]

        # 分批计算状态表示以节省内存
        phi_si_theta1 = []
        phi_sj_theta1 = []
        phi_si_theta2 = []
        phi_sj_theta2 = []
        
        sub_batch_size = 16  # 可以根据GPU内存调整
        for i in range(0, batch_size, sub_batch_size):
            end = min(i + sub_batch_size, batch_size)
            with torch.no_grad():
                # 第一个上下文
                phi_si_theta1.append(self.encoder(obs_i[i:end]))
                phi_sj_theta1.append(self.encoder(obs_j[i:end]))
                
                # 第二个上下文
                phi_si_theta2.append(self.encoder(augmentations.random_conv(obs_i[i:end].clone())))
                phi_sj_theta2.append(self.encoder(augmentations.random_conv(obs_j[i:end].clone())))
        
        # 合并子批次
        phi_si_theta1 = torch.cat(phi_si_theta1, dim=0)
        phi_sj_theta1 = torch.cat(phi_sj_theta1, dim=0)
        phi_si_theta2 = torch.cat(phi_si_theta2, dim=0)
        phi_sj_theta2 = torch.cat(phi_sj_theta2, dim=0)

        # 计算距离和损失
        d_sij = torch.norm(phi_si_theta1 - phi_sj_theta1, p=2, dim=1)
        
        bisim_loss, (L_BB, L_ICC, L_CC) = self.bisim_loss_fn(
            phi_si_theta1, phi_sj_theta1,
            phi_si_theta2, phi_sj_theta2,
            d_sij
        )

        return bisim_loss, (L_BB, L_ICC, L_CC)

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.args.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 计算双模拟损失
        bisim_loss, (L_BB, L_ICC, L_CC) = self.compute_bisim_loss(obs)
        
        # 合并损失
        total_loss = critic_loss + bisim_loss

        if L is not None:
            L.log('train_critic/critic_loss', critic_loss, step)
            L.log('train_critic/bisim_loss', bisim_loss, step)
            L.log('train_critic/L_BB', L_BB, step)
            L.log('train_critic/L_ICC', L_ICC, step)
            L.log('train_critic/L_CC', L_CC, step)
            L.log('train_critic/total_loss', total_loss, step)

        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.critic_optimizer.step()

        return {
            'critic_loss': critic_loss.item(),
            'bisim_loss': bisim_loss.item(),
            'L_BB': L_BB.item(),
            'L_ICC': L_ICC.item(),
            'L_CC': L_CC.item(),
            'total_loss': total_loss.item()
        }

    def update(self, replay_buffer, L, step):
        # 从replay buffer中采样，确保包含足够的状态对用于计算双模拟损失
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        # 更新critic并获取损失信息
        update_info = self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.args.actor_update_freq == 0:
            # 更新actor和alpha，并添加相关信息到update_info
            actor_info = self.update_actor_and_alpha(obs, L, step)
            if actor_info:
                update_info.update(actor_info)

        if step % self.args.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.args.critic_tau
            )

        return update_info

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            # 将LazyFrames转换为numpy数组
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.reshape(1, *obs.shape)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.reshape(1, 1, *obs.shape)  # 添加batch和channel维度
            
            obs = torch.FloatTensor(obs).to(self.device)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            # 检查动作维度
            action = mu.cpu().data.numpy().flatten()
            # 限制动作范围在[-1, 1]之间
            action = np.clip(action, -1.0, 1.0)
            # 只取前n个维度  n是环境动作空间的维度
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
            _, pi, _, _ = self.actor(obs, compute_log_pi=False)
            # 检查动作维度
            action = pi.cpu().data.numpy().flatten()
            # 限制动作范围在[-1, 1]之间
            action = np.clip(action, -1.0, 1.0)
            # 只取前n个维度  n是环境动作空间的维度
            action = action[:self.action_shape[0]]
            return action

    def update_actor_and_alpha(self, obs, L=None, step=None):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

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

        return {
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha_value': self.alpha.item()
        }