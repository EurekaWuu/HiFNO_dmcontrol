import sys
sys.path.append('/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/CLIP')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import algorithms.modules_multigpu as m
import augmentations
from algorithms.sac_multigpu import SAC
from bisimulation_loss_1 import BisimulationLoss

from algorithms.modules_multigpu import Actor, Critic
from algorithms.models.HiFNO_multigpu import HierarchicalFNO, PositionalEncoding, MultiScaleConv, PatchExtractor, Mlp, ConvResFourierLayer, SelfAttention, CrossAttentionBlock, TimeAggregator

import os
import time
from datetime import datetime
from PIL import Image
import torch.distributed as dist

import clip
from torchvision import transforms


_clip_model = None
_clip_preprocess = None

def get_clip_model(device):
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import clip
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        _clip_model = _clip_model.float()
    return _clip_model, _clip_preprocess

class HiFNOEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, args):
        super().__init__()
        
        self.out_dim = args.hidden_dim
        self.args = args
        
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
            num_scales=args.num_scales if hasattr(args, 'num_scales') else 3,
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

    def forward(self, obs, detach=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        
        # 输入维度
        if len(obs.shape) == 3:  # (C, H, W)
            obs = obs.unsqueeze(0)  # 添加batch维度
        elif len(obs.shape) == 2:  # (H, W)
            obs = obs.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        # 获取当前设备
        device = next(self.parameters()).device
        obs = obs.to(device)
        
        
        features = self.hifno(obs)  # (B, feature_dim, H', W')
        
        # 重塑为时序数据
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # (B, H, W, feature_dim)
        
        # 时间聚合
        features = features.unsqueeze(1)  # (B, 1, H, W, feature_dim)
        features = self.time_aggregator(features)  # (B, T', H, W, embed_dim)
        
        
        features = features.mean(dim=(1, 2, 3))  # (B, embed_dim)
        
        # 映射到所需的特征维度
        features = self.feature_map(features)  # (B, hidden_dim)
        
        # 检查最终输出维度
        if len(features.shape) > 2:
            features = features.mean(dim=1)  # 如果还有额外维度，取平均
        
        if detach:
            features = features.detach()
        
        return features

    def copy_conv_weights_from(self, source):
        pass


class CLIPClassifier:
    def __init__(self, descriptions, device):
        self.device = device
        
        self.model, self.preprocess = get_clip_model(device)
        
        self.descriptions = descriptions
        # 对所有描述进行编码
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(device)
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features = F.normalize(self.text_features, dim=1)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def classify_state(self, state_image):
        """对单个状态图像进行分类"""
        if not isinstance(state_image, torch.Tensor):
            state_image = torch.FloatTensor(state_image)
        
        # 确保图像是3通道的
        if state_image.shape[0] > 3:
            # 如果是堆叠的帧，只取最后一个帧
            state_image = state_image[-3:]
        
        image = self.transform(state_image.cpu()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        
        class_idx = similarities.argmax().item()
        
        return class_idx, similarities[0]
    
    def batch_classify_states(self, state_images):
        """对一批状态图像进行分类"""
        if not isinstance(state_images, torch.Tensor):
            state_images = torch.FloatTensor(state_images)
        
        batch_size = state_images.shape[0]
        
        # 确保图像是3通道的
        if state_images.shape[1] > 3:
            # 如果是堆叠的帧，只取最后一个帧
            state_images = state_images[:, -3:]
        
        processed_images = []
        for i in range(batch_size):
            processed_image = self.transform(state_images[i].cpu())
            processed_images.append(processed_image)
        
        processed_images = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(processed_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        
        class_indices = similarities.argmax(dim=-1)
        
        return class_indices, similarities


class HiFNOBisimAgent(SAC, nn.Module):
    def __init__(self, obs_shape, action_shape, args):
        nn.Module.__init__(self)
        
        self.args = args
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        
        self.action_shape = action_shape
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if hasattr(args, 'work_dir'):
            args.work_dir = os.path.join(args.work_dir, f'{timestamp}')
            if not os.path.exists(args.work_dir):
                os.makedirs(args.work_dir)
        
        
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

        # 初始化语义类内一致性损失函数
        self.bisim_loss_fn = BisimulationLoss(
            lambda_SC=getattr(args, 'lambda_SC', 0.5),
            lambda_clip=getattr(args, 'lambda_clip', 0.5),
            p=getattr(args, 'bisim_p', 2),
            use_sc_loss=getattr(args, 'use_sc_loss', True),
            use_clip_bisim_loss=getattr(args, 'use_clip_bisim_loss', True)
        )
        
        self.descriptions = [
            "The mannequin's torso is upright with a slight forward lean, while one leg is extended behind for maximum push-off, and the other leg is lifting towards the front for the next step.",
            "In this stride, the mannequin's torso remains straight and stable, with one leg bent at the knee in mid-air while the other leg drives forward, ensuring efficient movement and balance.",
            "With its torso aligned, the mannequin is lifting one leg in front of the body while the other leg extends backward, demonstrating an efficient running posture with proper leg extension and cadence.",
            "The mannequin maintains an upright posture, with its knees bent and one leg pushing off the ground while the other leg moves toward the front, ensuring smooth, rhythmic strides.",
            "The torso is held firm and steady, while one leg is fully extended behind, and the other leg is bent, bringing the knee high to prepare for the next stride in a fluid running motion.",
            "As the mannequin runs, its torso stays balanced and straight, with alternating leg movements—one leg propels the body forward while the other leg swings back to maximize speed and efficiency."
        ]
        
        self.clip_loss_weight = getattr(args, 'clip_loss_weight', 0.5)

    def compute_clip_guided_loss(self, obs, actions):
        """计算CLIP引导的损失，使用整个批次一次性处理"""
        device = next(self.parameters()).device
        

        if not hasattr(self, 'clip_classifier'):
            self.clip_classifier = CLIPClassifier(self.descriptions, device=device)
        

        with torch.cuda.amp.autocast(enabled=False):  # 启用混合精度
            state_features = self.encoder(obs)
        
        # 获取最后一帧作为当前状态
        if obs.shape[1] > 3:  # 如果是堆叠的帧
            frame_size = obs.shape[1] // self.args.frame_stack
            current_frame = obs[:, -frame_size:].clone()
        else:
            current_frame = obs.clone()
        
        # 使用CLIP分类器对状态进行分类
        class_indices, similarities = self.clip_classifier.batch_classify_states(current_frame)
        
        total_loss, (sc_loss, clip_bisim_loss) = self.bisim_loss_fn.compute_total_loss(
            state_features, actions, similarities, class_indices
        )
        
        return total_loss, class_indices, similarities, (sc_loss, clip_bisim_loss)

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)


        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)


        clip_loss, class_indices, similarities, (sc_loss, clip_bisim_loss) = self.compute_clip_guided_loss(obs, action)
        

        total_loss = critic_loss + self.clip_loss_weight * clip_loss

        if L is not None:
            L.log('train_critic/loss', critic_loss, step)
            L.log('train_critic/clip_loss', clip_loss, step)
            L.log('train_critic/sc_loss', sc_loss, step)
            L.log('train_critic/clip_bisim_loss', clip_bisim_loss, step)
            L.log('train_critic/total_loss', total_loss, step)

        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.critic_optimizer.step()

        return {
            'critic_loss': critic_loss.item(),
            'clip_loss': clip_loss.item(),
            'sc_loss': sc_loss.item(),
            'clip_bisim_loss': clip_bisim_loss.item(),
            'total_loss': total_loss.item()
        }

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        # 将numpy数组转换为tensor
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

        device = next(self.parameters()).device
        obs = obs.to(device)
        next_obs = next_obs.to(device)
        action = action.to(device)
        reward = reward.to(device)
        not_done = not_done.to(device)

        update_info = self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            actor_info = self.update_actor_and_alpha(obs, L, step)
            if actor_info:
                update_info.update(actor_info)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_tau
            )

        return update_info

    def update_actor_and_alpha(self, obs, L=None, step=None):

        device = next(self.parameters()).device

        _, pi, log_pi, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

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
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        
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
        
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # 将观测转换为tensor
            if not isinstance(obs, torch.Tensor):
                obs = torch.FloatTensor(obs)
            
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.unsqueeze(0)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
            
            obs = obs.to(device)
            
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            # 检查动作维度
            action = mu.cpu().data.numpy().flatten()
            action = np.clip(action, -1.0, 1.0)
            action = action[:self.action_shape[0]]
            
            return action

    def sample_action(self, obs):
        with torch.no_grad():

            device = next(self.parameters()).device
            
            # 将观测转换为tensor
            if not isinstance(obs, torch.Tensor):
                obs = torch.FloatTensor(obs)
            
            # 处理维度
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.unsqueeze(0)  # 添加batch维度
            elif len(obs.shape) == 2:  # (H, W)
                obs = obs.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
            
            obs = obs.to(device)
            
            _, pi, _, _ = self.actor(obs, compute_log_pi=False)
            # 检查动作维度
            action = pi.cpu().data.numpy().flatten()
            action = np.clip(action, -1.0, 1.0)
            action = action[:self.action_shape[0]]
            
            return action