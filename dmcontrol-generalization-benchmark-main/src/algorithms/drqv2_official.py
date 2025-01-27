# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re
import time
from torch import distributions as pyd
import io
import datetime
import traceback
from collections import defaultdict
from torch.utils.data import IterableDataset


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        # 线性变化linear(init, final, duration)
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        # 分段线性step_linear(init, final1, dur1, final2, dur2)
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2

    raise NotImplementedError(schdl)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        # x - x.detach()保留梯度通道
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

    @property
    def mean(self):
        return self._clamp(self.loc)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w, "只支持方形图像，要额外处理。"
        padding = (self.pad, self.pad, self.pad, self.pad)
        x = F.pad(x, padding, mode='replicate')

        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad,
                                device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2),
            device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift

        return F.grid_sample(
            x, grid, padding_mode='zeros', align_corners=False
        )


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) == 3
        # DrQ默认会将84x84图像卷积得到32x35x35作为中间特征
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU()
        )

        self.apply(weight_init)

    def forward(self, obs):
        # 图像像素值缩放到[-0.5, 0.5]
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.size(0), -1)
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
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        # tanh限制动作范围 [-1,1]
        mu = torch.tanh(mu)
        # std保持为常数张量
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.action_shape = action_shape  # 保存动作形状参数
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
        # 确保动作维度正确
        assert action.shape[1] == self.action_shape[0], f"Expected action dimension {self.action_shape[0]}, but got {action.shape[1]}"
        h_action = torch.cat([h, action], dim=-1)
        # print(f"Dimension of h_action before Q1 and Q2: {h_action.shape}")  # 注释掉打印维度信息
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class ReplayBufferStorage:
    def __init__(self, replay_dir):
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def add(self, obs, action, reward, next_obs, done):
        self._current_episode['observation'].append(obs)
        self._current_episode['action'].append(action)
        self._current_episode['reward'].append(reward)
        self._current_episode['discount'].append(1.0 - done)
        
        if done:
            episode = dict()
            for k, v in self._current_episode.items():
                episode[k] = np.array(v)
            episode['observation'] = np.concatenate([
                episode['observation'], 
                next_obs[None]], axis=0)
            
            self._store_episode(episode)
            self._current_episode = defaultdict(list)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class DrQReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._storage = ReplayBufferStorage(replay_dir)

    def add(self, obs, action, reward, next_obs, done):
        self._storage.add(obs, action, reward, next_obs, done)

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        last_warning_time = 0  # 控制警告输出频率
        warning_interval = 1  # 秒
        
        while True:
            # 确保有数据可用
            if not self._episode_fns:
                self._try_fetch()
                if not self._episode_fns:  
                    current_time = time.time()
                    if current_time - last_warning_time > warning_interval:
                        # print("[DrQReplayBuffer] No episodes available, waiting for data...")
                        last_warning_time = current_time
                    # 返回一个全零的批次
                    yield (
                        np.zeros((1, 9, 84, 84), dtype=np.uint8),  # obs
                        np.zeros((1, 6), dtype=np.float32),        # action
                        np.zeros(1, dtype=np.float32),             # reward
                        np.ones(1, dtype=np.float32),              # discount
                        np.zeros((1, 9, 84, 84), dtype=np.uint8)   # next_obs
                    )
                    continue
            
            try:
                yield self._sample()
            except Exception as e:
                print(f"[DrQReplayBuffer] Error in sampling: {str(e)}")
                self._try_fetch()

def make_drq_replay_loader(replay_dir, max_size, batch_size, num_workers,
                           save_snapshot, nstep, discount):
    iterable = DrQReplayBuffer(
        replay_dir=replay_dir,
        max_size=max_size,
        num_workers=num_workers,
        nstep=nstep,
        discount=discount,
        fetch_every=1000,
        save_snapshot=save_snapshot
    )
    
    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=None
    )
    return loader


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class DrQV2OffAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        batch_size,
        num_seed_frames,
        action_repeat
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.batch_size = batch_size
        self.num_seed_frames = num_seed_frames
        self.action_repeat = action_repeat

        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)

        self.critic = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

        self.seed_until_step = Until(self.num_seed_frames, self.action_repeat)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode=False):
        """
        eval_mode=True 输出均值动作（评估时无随机性）
        eval_mode=False 从分布中采样动作
        若step < num_expl_steps时使用随机探索
        """
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)  
        if len(obs.shape) == 3:
            obs = obs.reshape(1, *obs.shape)       # (C,H,W) => (B=1,C,H,W)
        elif len(obs.shape) == 2:
            obs = obs.reshape(1, 1, *obs.shape)    # (H,W) => (B=1,C=1,H,W)
        elif len(obs.shape) == 1:
            obs = obs.reshape(1, *obs.shape)       # (D,) => (B=1,D)

        obs = torch.FloatTensor(obs).to(self.device)

        obs = self.encoder(obs)
        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.detach().cpu().numpy()[0]

    def select_action(self, obs):
        """
        兼容旧接口select_action(obs) 
        默认使用评估模式，输出动作均值
        """
        # 如果要评估阶段也有某种随机性，调整step或eval_mode
        return self.act(obs, step=0, eval_mode=True)

    def sample_action(self, obs):
        """
        兼容旧接口sample_action(obs)
        默认使用训练模式，从分布中采样
        """
        return self.act(obs, step=0, eval_mode=False)

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        stddev = schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        # DrQ-v2直接最大化Q
        actor_loss = -Q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()

        return metrics

    def update(self, replay_iter, L, step):
        metrics = dict()
        
        if step % self.update_every_steps != 0:
            return metrics
        
        # 检查是否已收集足够的初始数据
        if self.seed_until_step(step):
            if step % 1000 == 0:  
                print(f"[Step {step}] Collecting initial data, not updating yet...")
            return metrics
        
        try:
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs = batch
            
            # # 4. 打印调试信息（每1000步打印一次）
            # if step % 1000 == 0:
            #     print(f"[DrQV2Off] Step {step} - Batch shapes:")
            #     print(f"- obs: {obs.shape}")
            #     print(f"- action: {action.shape}")
            #     print(f"- reward: {reward.shape}")
            #     print(f"- discount: {discount.shape}")
            #     print(f"- next_obs: {next_obs.shape}")
            
            # 检查数据是否是空的占位数据
            if isinstance(obs, np.ndarray) and obs.size > 0:
                if np.all(obs == 0) and np.all(action == 0):
                    if step % 1000 == 0:
                        print("[DrQV2Off] Received dummy data, skipping update")
                    return metrics
            
            # 正常的更新流程
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs.astype(np.float32)).to(self.device)
                next_obs = torch.FloatTensor(next_obs.astype(np.float32)).to(self.device)
            else:
                obs = obs.float().to(self.device)
                next_obs = next_obs.float().to(self.device)
            
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            discount = torch.FloatTensor(discount).to(self.device)
            
            # 移除多余的维度
            obs = obs.squeeze(1)
            action = action.squeeze(1)
            next_obs = next_obs.squeeze(1)
            
            # if step % 1000 == 0:
            #     print(f"After squeeze - Batch shapes:")
            #     print(f"- obs: {obs.shape}")
            #     print(f"- action: {action.shape}")
            #     print(f"- next_obs: {next_obs.shape}")
            
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)
            
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

            metrics.update(
                self.update_critic(obs, action, reward, discount, next_obs, step)
            )
            metrics.update(self.update_actor(obs.detach(), step))

            soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
            
            if self.use_tb:
                for k, v in metrics.items():
                    L.log(f'train/{k}', v, step)
            
        except Exception as e:
            if step % 1000 == 0:  
                print(f"[DrQV2Off] Error in update: {str(e)}")

        return metrics

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode