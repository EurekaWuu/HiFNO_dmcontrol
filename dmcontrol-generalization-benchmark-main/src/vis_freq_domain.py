import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from datetime import datetime

from env.wrappers import make_env
from algorithms.hifno import HiFNOEncoder, HiFNOAgent
from algorithms.models.HiFNO import HierarchicalFNO
import utils


FRAME_INDICES = [0, 200, 600,800]

ENV_DOMAIN = "walker"
ENV_TASK = "walk"

MODEL_NAME = "svea_walker_walk_20241014_224006"
MODEL_STEP = "500000"
MODEL_PATH = f"logs/{MODEL_NAME}/{ENV_DOMAIN}_{ENV_TASK}/svea/1/model/{MODEL_STEP}.pt"


class Args:
    def __init__(self):
        self.frame_stack = 3
        self.hidden_dim = 256
        self.embed_dim = 128
        self.depth = 2
        self.num_scales = 3

def create_env(domain_name=ENV_DOMAIN, task_name=ENV_TASK):
    env = make_env(
        domain_name=domain_name,
        task_name=task_name,
        seed=42,
        episode_length=1000,
        action_repeat=1,
        image_size=84,
        mode='train'
    )
    return env

def compute_fft_2d(image):
    fft = torch.fft.fft2(image)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shift)
    return magnitude

def visualize_fno_features(encoder, obs):
    if not isinstance(obs, torch.Tensor):
        obs = torch.FloatTensor(obs)
    if len(obs.shape) == 3:
        obs = obs.unsqueeze(0)
    
    obs = obs.to(encoder.device)
    
    with torch.no_grad():
        features = encoder.hifno(obs, return_intermediate=True)
    
    return features

def plot_feature_maps(features, num_channels_to_show=4, save_path='feature_maps.png'):
    num_scales = len(features)
    fig, axes = plt.subplots(num_scales, num_channels_to_show + 1, figsize=(15, 3*num_scales))
    
    for scale_idx, scale_features in enumerate(features):
        
        fft_magnitude = compute_fft_2d(scale_features[0])
        
        
        avg_magnitude = torch.mean(fft_magnitude, dim=0)
        axes[scale_idx, 0].imshow(torch.log(avg_magnitude + 1).cpu())
        axes[scale_idx, 0].set_title(f'Scale {scale_idx} Avg FFT')
        
        # 绘制选定通道的特征图
        for i in range(num_channels_to_show):
            if i < scale_features.shape[1]:
                axes[scale_idx, i+1].imshow(scale_features[0, i].cpu())
                axes[scale_idx, i+1].set_title(f'Channel {i}')
    
    plt.tight_layout()
    plt.savefig(save_path)  
    plt.close()  

def analyze_observation(obs, save_path='observation_fft.png'):
    obs_tensor = torch.FloatTensor(obs)
    
    fft_magnitudes = []
    for c in range(obs.shape[0]):
        magnitude = compute_fft_2d(obs_tensor[c])
        fft_magnitudes.append(magnitude)
    
    fig, axes = plt.subplots(2, obs.shape[0], figsize=(15, 6))
    
    for i in range(obs.shape[0]):
        axes[0, i].imshow(obs[i])
        axes[0, i].set_title(f'Channel {i}')
        
        axes[1, i].imshow(torch.log(fft_magnitudes[i] + 1).numpy())
        axes[1, i].set_title(f'Channel {i} FFT')
    
    plt.tight_layout()
    plt.savefig(save_path)  
    plt.close()  

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def collect_frames(env, frame_indices=[0, 10, 20]):
    obs = env.reset()
    observations = []  # 不自动保存第0帧
    
    max_frame = max(frame_indices)
    for i in range(max_frame + 1):
        if i in frame_indices:  
            observations.append(np.array(obs))
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
    
    return observations

def analyze_multiple_frames(observations, save_path='observation_fft.png'):
    num_frames = len(observations)
    
    fig, axes = plt.subplots(2, num_frames * 3, figsize=(5 * num_frames, 8))
    
    for frame_idx, obs in enumerate(observations):
        obs_tensor = torch.FloatTensor(obs)
        
        fft_magnitudes = []
        for c in range(obs.shape[0]):
            magnitude = compute_fft_2d(obs_tensor[c])
            fft_magnitudes.append(magnitude)
        
        for i in range(3):  
            channel_idx = frame_idx * 3 + i
            
            
            axes[0, channel_idx].imshow(obs[i])
            axes[0, channel_idx].set_title(f'Frame {frame_idx} Ch {i}')
            
            
            axes[1, channel_idx].imshow(torch.log(fft_magnitudes[i] + 1).numpy())
            axes[1, channel_idx].set_title(f'Frame {frame_idx} Ch {i} FFT')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_total_frames(env):
    episode_length = env._max_episode_steps 
    action_repeat = env.action_repeat if hasattr(env, 'action_repeat') else 1 
    total_frames = episode_length // action_repeat 
    return total_frames

def load_agent(model_path):
    print(f"Loading model from: {model_path}")
    agent = torch.load(model_path)
    agent.eval()  
    return agent

def collect_frames_with_agent(env, frame_indices, agent):
    obs = env.reset()
    observations = []
    
    max_frame = max(frame_indices)
    for i in range(max_frame + 1):
        if i in frame_indices:
            observations.append(np.array(obs))
        
        with torch.no_grad():
            action = agent.select_action(obs)
        obs, _, _, _ = env.step(action)
    
    return observations

def main():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(current_dir, 'visualization_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory at: {output_dir}")
    

    model_path = os.path.join(current_dir, MODEL_PATH)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at: {model_path}")
        return
        
    agent = load_agent(model_path)
    print("Model loaded successfully")
    
    timestamp = get_timestamp()
    args = Args()
    

    env = create_env()  
    
    total_frames = get_total_frames(env)
    print(f"Environment settings:")
    print(f"- Episode length: {env._max_episode_steps}")
    print(f"- Action repeat: {env.action_repeat if hasattr(env, 'action_repeat') else 1}")
    print(f"- Total available frames: {total_frames}")
    print(f"- Current frame indices for analysis: {FRAME_INDICES}")
    
    if max(FRAME_INDICES) >= total_frames:
        print(f"Warning: Some frame indices exceed the total number of frames!")
        print(f"Please select frame indices between 0 and {total_frames-1}")
        return
    
    
    observations = collect_frames_with_agent(env, FRAME_INDICES, agent)
    print(f"Collected frames at indices: {FRAME_INDICES}")
    
    obs_save_path = os.path.join(output_dir, f'observation_fft_frames_{timestamp}.png')
    analyze_multiple_frames(observations, save_path=obs_save_path)
    print(f"Saved multi-frame analysis to {obs_save_path}")
    
    encoder = HiFNOEncoder(
        obs_shape=observations[0].shape,
        feature_dim=args.embed_dim,
        args=args
    )
    
    features = visualize_fno_features(encoder, observations[0])
    feature_save_path = os.path.join(output_dir, f'feature_maps_{timestamp}.png')
    plot_feature_maps(features, save_path=feature_save_path)
    print(f"Saved feature maps to {feature_save_path}")

if __name__ == "__main__":
    main()




'''

python /mnt/lustre/GPU4/home/wuhanpeng/dmcontrol-generalization-benchmark-main/dmcontrol-generalization-benchmark-main/src/vis_freq_domain.py


'''