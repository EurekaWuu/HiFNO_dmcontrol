import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime
import re
import torchvision.transforms as TF
import torchvision.datasets as datasets
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.distributions as pyd
from torch.distributions.utils import _standard_normal
import time
import torch.nn as nn
import torch.nn.functional as F
import argparse

places_dataloader = None
places_iter = None


def schedule(schedule_str, step):
    """
    调度模式：
    常数值: 直接返回该数值
    linear(a,b,n): 从a线性变化到b，用时n步
    step_linear(init,final1,duration1,final2,duration2): 两段式线性变化
    """
    try:
        return float(schedule_str)
    except ValueError:
        # 原始linear模式
        match = re.match(r'linear\(([\d.]+),([\d.]+),(\d+)\)', schedule_str)
        if match:
            start, end, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return start + (end - start) * mix
            
        # step_linear模式
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schedule_str)
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
                
        raise ValueError(f'Invalid schedule format: {schedule_str}')


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        'timestamp': str(datetime.now()),
        # 'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
        'args': vars(args)
    }
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(current_dir, 'setup', 'config.cfg')
    
    try:
        with open(path) as f:
            data = json.load(f)
        if key is not None:
            return data[key]
        return data
    except FileNotFoundError:
        default_config = {
            "datasets": [
                "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/datasets/DAVIS",
                "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/datasets/places365_standard"
            ]
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        if key is not None:
            return default_config[key]
        return default_config


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
    fpath = os.path.join(dir_path, f'*.{filetype}')
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True):
        self.capacity = capacity
        self.batch_size = batch_size

        self._obses = []
        if prefill:
            self._obses = prefill_memory(self._obses, capacity, obs_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(
            0, self.capacity if self.full else self.idx, size=n
        )

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def sample_soda(self, n=None):
        idxs = self._get_idxs(n)
        obs, _ = self._encode_obses(idxs)
        return torch.as_tensor(obs).cuda().float()

    def __sample__(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones

    def sample_curl(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        pos = augmentations.random_crop(obs.clone())
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones, pos

    def sample_drq(self, n=None, pad=4):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_shift(obs, pad)
        next_obs = augmentations.random_shift(next_obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample_svea(self, n=None, pad=4):
        return self.sample_drq(n=n, pad=pad)

    def sample(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0]//3

    def frame(self, i):
        return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f'{count:,}'


class BisimReplayBuffer(ReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, batch_size):
        super().__init__(obs_shape, action_shape, capacity, batch_size)
        
    def sample(self):
        """分批处理数据以减少内存使用"""
        # 第一批
        idxs1 = np.random.randint(
            0, self.capacity if self.full else self.idx, 
            size=self.batch_size
        )
        obs1, next_obs1 = self._encode_obses(idxs1)
        
        # 第二批
        idxs2 = np.random.randint(
            0, self.capacity if self.full else self.idx, 
            size=self.batch_size
        )
        obs2, next_obs2 = self._encode_obses(idxs2)
        
        # 合并数据
        obs = np.concatenate([obs1, obs2], axis=0)
        next_obs = np.concatenate([next_obs1, next_obs2], axis=0)
        
        # 获取其他数据
        idxs = np.concatenate([idxs1, idxs2])
        
        # 转换为张量
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()
        
        return obs, actions, rewards, next_obs, not_dones


def _load_places(batch_size=256, image_size=84, num_workers=8, use_val=False):
    global places_dataloader, places_iter
    partition = 'val' if use_val else 'train'
    print(f'Loading {partition} partition of places365_standard...')
    for data_dir in load_config('datasets'):
        if os.path.exists(data_dir):
            fp = os.path.join(data_dir, 'places365_standard', partition)
            if not os.path.exists(fp):
                print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
                fp = data_dir
            places_dataloader = torch.utils.data.DataLoader(
                datasets.ImageFolder(fp, TF.Compose([
                    TF.RandomResizedCrop(image_size),
                    TF.RandomHorizontalFlip(),
                    TF.ToTensor()
                ])),
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True)
            places_iter = iter(places_dataloader)
            break
    if places_iter is None:
        raise FileNotFoundError('failed to find places365 data at any of the specified paths')
    print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
    global places_iter
    try:
        imgs, _ = next(places_iter)
        if imgs.size(0) < batch_size:
            places_iter = iter(places_dataloader)
            imgs, _ = next(places_iter)
    except StopIteration:
        places_iter = iter(places_dataloader)
        imgs, _ = next(places_iter)
    return imgs.cuda()


def random_overlay(x, dataset='places365_standard'):
    """Randomly overlay an image from Places"""
    global places_iter
    alpha = 0.5

    if dataset == 'places365_standard':
        if places_dataloader is None:
            _load_places(batch_size=x.size(0), image_size=x.size(-1))
        imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
    else:
        raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

    return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


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


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')