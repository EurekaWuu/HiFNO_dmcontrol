# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

import utils
from algorithms.modules import weight_init


class TruncatedNormal(td.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.loc = loc
        self.scale = scale

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = eps * self.scale
        
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
            
        x = self.loc + eps
        return self._clamp(x)

    def log_prob(self, value):
        return super().log_prob(value)

    @property
    def mean(self):
        return self._clamp(self.loc)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(
            x,
            grid,
            padding_mode='zeros',
            align_corners=False
        )


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU()
        )

        self.apply(weight_init)

    def forward(self, obs):
        # 将图像像素值缩放到 [-0.5, 0.5]
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.apply(weight_init)

    def forward(self, obs, std):
        """输出给定 obs 的策略分布(dist)，分布为 TruncatedNormal"""
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)  # 动作限幅在 [-1, 1]
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std, low=-1.0, high=1.0, eps=1e-6)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class DrQV2Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        args
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.critic_target_tau = args.critic_target_tau
        self.update_every_steps = args.update_every_steps
        self.use_tb = args.use_tb
        self.num_expl_steps = args.num_expl_steps
        self.stddev_schedule = args.stddev_schedule
        self.stddev_clip = args.stddev_clip
        self.min_std = 1e-4  
        self._step = 0  
        lr = args.lr
        feature_dim = args.feature_dim
        hidden_dim = args.hidden_dim

        
        self.encoder = Encoder(obs_shape).to(self.device)
        self.actor = Actor(
            self.encoder.repr_dim,
            action_shape,
            feature_dim,
            hidden_dim
        ).to(self.device)

        self.critic = Critic(
            self.encoder.repr_dim,
            action_shape,
            feature_dim,
            hidden_dim
        ).to(self.device)
        self.critic_target = Critic(
            self.encoder.repr_dim,
            action_shape,
            feature_dim,
            hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)


        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(action_shape[0])

        
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

        self.episode_reward = 0  

    @property
    def alpha(self):
        # alpha = exp(log_alpha)
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def select_action(self, obs):
        with torch.no_grad():
            
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.reshape(1, *obs.shape)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.reshape(1, 1, *obs.shape)  # 添加batch和channel维度

            obs = torch.FloatTensor(obs).to(self.device)
            obs = self.encoder(obs)
            stddev = self.min_std  
            dist = self.actor(obs, stddev)
            return dist.mean.cpu().numpy()[0]

    def sample_action(self, obs):
        with torch.no_grad():
            
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.reshape(1, *obs.shape)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.reshape(1, 1, *obs.shape)  # 添加batch和channel维度

            obs = torch.FloatTensor(obs).to(self.device)
            obs = self.encoder(obs)
            stddev = utils.schedule(self.stddev_schedule, self._step)
            dist = self.actor(obs, stddev)
            action = dist.sample(clip=self.stddev_clip)
            if self._step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
            return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # 计算目标 Q
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        # 当前 Q
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['train/critic_loss'] = critic_loss.item()
            metrics['train/critic_q1'] = Q1.mean().item()
            metrics['train/critic_q2'] = Q2.mean().item()
            metrics['train/critic_target_q'] = target_Q.mean().item()

        
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # (B,1)

        
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        
        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.log_alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_opt.step()

        
        if self.use_tb:
            metrics['train/actor_loss'] = actor_loss.item()
            metrics['train/actor_logprob'] = log_prob.mean().item()
            metrics['train/actor_entropy'] = -log_prob.mean().item()
            metrics['train/alpha_loss'] = alpha_loss.item()
            metrics['train/alpha_value'] = self.alpha.item()

        return metrics

    def update(self, replay_buffer, L, step):
        self._step = step
        if step % self.update_every_steps != 0:
            return

        batch = replay_buffer.sample_drq()
        obs, action, reward, next_obs, not_done = batch
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        discount = not_done  # not_done 作为折扣因子

        
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        
        critic_metrics = self.update_critic(obs, action, reward, discount, next_obs, step)
        if self.use_tb:
            for k, v in critic_metrics.items():
                L.log(k, v, step)

        
        actor_metrics = self.update_actor(obs.detach(), step)
        if self.use_tb:
            for k, v in actor_metrics.items():
                L.log(k, v, step)

        
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
