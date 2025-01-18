# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from torchvision import transforms
import utils
from utils import random_overlay
from algorithms.modules import weight_init
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import random

class Places365Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, file_list, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

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
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()
        self.model = resnet18(pretrained=True)
        
        # 修改第一层卷积，接受 9 通道输入
        self.model.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224)
            ])

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.repr_dim = 1024
        self.image_channel = 9  # 修改为 9 通道
        x = torch.randn([32] + [9, 84, 84])
        with torch.no_grad():
            out_shape = self.forward_conv(x).shape
        self.out_dim = out_shape[1]
        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)

    @torch.no_grad()
    def forward_conv(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == 'layer2':
                break
        
        if flatten:
            obs = obs.view(obs.size(0), -1)
        return obs

    def forward(self, obs):
        conv = self.forward_conv(obs)
        out = self.fc(conv)
        out = self.ln(out)
        return out




class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                 nn.LayerNorm(feature_dim), 
                                 nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim, action_shape[0]))

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std, low=-1.0, high=1.0, eps=1e-6)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class PIEG:
    def __init__(self, obs_shape, action_shape, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.critic_target_tau = args.critic_target_tau
        self.update_every_steps = args.update_every_steps
        self.use_tb = args.use_tb
        self.num_expl_steps = args.num_expl_steps
        self.stddev_schedule = args.stddev_schedule
        self.stddev_clip = args.stddev_clip
        self._step = 0
        
        # models
        self.encoder = ResEncoder().to(self.device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, 
                         args.feature_dim, args.hidden_dim).to(self.device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, 
                           args.feature_dim, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                  args.feature_dim, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

        
        self.places_dataloader = None
        self.places_iter = None
        self._init_places_loader(batch_size=32, image_size=84)

    def _init_places_loader(self, batch_size=32, image_size=84):
        from utils import load_config
        from PIL import Image
        
        for data_dir in load_config('datasets'):
            if not os.path.exists(data_dir):
                continue
            
            if 'places365_standard' in data_dir:
                train_file = os.path.join(data_dir, 'train.txt')
                if not os.path.exists(train_file):
                    continue
                    
                try:
                    with open(train_file, 'r') as f:
                        image_list = [line.strip() for line in f.readlines()]
                    print(f"Found {len(image_list)} images in train.txt")
                    
                    
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()
                    ])
                    dataset = Places365Dataset(data_dir, image_list, transform=transform)
                    
                    self.places_dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True
                    )
                    self.places_iter = iter(self.places_dataloader)
                    print(f'Successfully loaded Places365 dataset from {data_dir}')
                    return
                    
                except Exception as e:
                    print(f'Error loading Places365 dataset: {str(e)}')
                    continue
        
        raise FileNotFoundError('Failed to load Places365 dataset')

    def _get_places_batch(self, batch_size):
        try:
            imgs, _ = next(self.places_iter)
            if imgs.size(0) < batch_size:
                self.places_iter = iter(self.places_dataloader)
                imgs, _ = next(self.places_iter)
        except (StopIteration, AttributeError):
            self.places_iter = iter(self.places_dataloader)
            imgs, _ = next(self.places_iter)
        return imgs.cuda()

    def _random_overlay(self, x):
        """在图像上随机叠加 Places365 图像"""
        alpha = 0.5
        # 如果当前batch size与数据加载器不匹配，重新初始化数据加载器
        if self.places_dataloader is None or self.places_dataloader.batch_size != x.size(0):
            self._init_places_loader(batch_size=x.size(0), image_size=x.size(-1))
        imgs = self._get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
        return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        if eval_mode:
            return self.select_action(obs)
        else:
            return self.sample_action(obs)

    def update_critic(self, obs, action, reward, discount, next_obs, step, aug_obs):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)

        critic_loss = 0.5 * (critic_loss + aug_loss)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
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
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_buffer, logger, step):
        self._step = step
        
        if step % self.update_every_steps != 0:
            return
        
        batch = replay_buffer.sample()
        obs, action, reward, next_obs, not_done = batch
        
        
        obs = self.aug(obs.float())
        original_obs = obs.clone()
        next_obs = self.aug(next_obs.float())
        
        
        obs = self.encoder(obs)
        aug_obs = self.encoder(self._random_overlay(original_obs))
        
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
        
        
        metrics = self.update_critic(obs, action, reward, 
                                   not_done, next_obs, step, aug_obs)
        
        
        metrics.update(self.update_actor(obs.detach(), step))
        
        
        utils.soft_update_params(self.critic, self.critic_target,
                               self.critic_target_tau)
                           
        
        if logger is not None:
            for k, v in metrics.items():
                logger.log(f'train/{k}', v, step)

    def select_action(self, obs):
        with torch.no_grad():
            # 处理输入维度
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.reshape(1, *obs.shape)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.reshape(1, 1, *obs.shape)  # 添加batch和channel维度

            obs = torch.FloatTensor(obs).to(self.device)
            obs = self.encoder(obs)  # 不需要unsqueeze，已经添加了batch维度
            # 评估时使用确定性策略
            dist = self.actor(obs, std=0.0)
            return dist.mean.cpu().numpy()[0]

    def sample_action(self, obs):
        with torch.no_grad():
            # 处理输入维度
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.reshape(1, *obs.shape)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.reshape(1, 1, *obs.shape)  # 添加batch和channel维度

            obs = torch.FloatTensor(obs).to(self.device)
            obs = self.encoder(obs)  # 不需要unsqueeze，已经添加了batch维度
            dist = self.actor(obs, std=utils.schedule(self.stddev_schedule, self._step))
            action = dist.sample(clip=self.stddev_clip)
            if self._step < self.num_expl_steps:  # 添加探索步骤
                action.uniform_(-1.0, 1.0)
            return action.cpu().numpy()[0]
