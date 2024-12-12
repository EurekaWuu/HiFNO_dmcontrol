import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

class SVEA_VIS(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.svea_alpha = args.svea_alpha
        self.svea_beta = args.svea_beta
        
        # 初始化可视化工具
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'runs/SVEA_experiment_{timestamp}')
        self.action_shape = action_shape
        
        # 创建图表风格
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def visualize_critic_values(self, current_Q1, current_Q2, target_Q, step):
        """可视化Q值分布"""
        plt.figure(figsize=(12, 6))
        
        # Q值分布直方图
        plt.subplot(1, 2, 1)
        sns.histplot(data=current_Q1.detach().cpu().numpy(), label='Q1', alpha=0.6)
        sns.histplot(data=current_Q2.detach().cpu().numpy(), label='Q2', alpha=0.6)
        plt.title('Q-Values Distribution')
        plt.legend()
        
        # Q值随时间变化
        plt.subplot(1, 2, 2)
        plt.plot([current_Q1.mean().item(), current_Q2.mean().item(), target_Q.mean().item()], 
                label=['Q1 Mean', 'Q2 Mean', 'Target Q Mean'])
        plt.title('Q-Values Mean')
        
        self.writer.add_figure('Critic/Q_values', plt.gcf(), step)
        plt.close()

    def visualize_action_distribution(self, actions, step):
        """可视化动作分布"""
        actions_np = actions.detach().cpu().numpy()
        
        fig = plt.figure(figsize=(12, 4 * ((self.action_shape[0]-1)//3 + 1)))
        for i in range(self.action_shape[0]):
            plt.subplot(((self.action_shape[0]-1)//3 + 1), 3, i+1)
            sns.kdeplot(data=actions_np[:, i])
            plt.title(f'Action Dimension {i}')
            plt.xlabel('Action Value')
            plt.ylabel('Density')
            
        plt.tight_layout()
        self.writer.add_figure('Actions/distribution', plt.gcf(), step)
        plt.close()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        if self.svea_alpha == self.svea_beta:
            obs = utils.cat(obs, augmentations.random_conv(obs.clone()))
            action = utils.cat(action, action)
            target_Q = utils.cat(target_Q, target_Q)

            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = (self.svea_alpha + self.svea_beta) * \
                (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
        else:
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = self.svea_alpha * \
                (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

            obs_aug = augmentations.random_conv(obs.clone())
            current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
            critic_loss += self.svea_beta * \
                (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

        # 记录训练指标
        if L is not None and step is not None:
            L.log('train_critic/loss', critic_loss, step)
            self.writer.add_scalar('Critic/Loss', critic_loss.item(), step)
            
            # 每1000步进行一次详细可视化
            if step % 1000 == 0:
                self.visualize_critic_values(current_Q1, current_Q2, target_Q, step)
                self.visualize_action_distribution(action, step)
            
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()
        
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target() 