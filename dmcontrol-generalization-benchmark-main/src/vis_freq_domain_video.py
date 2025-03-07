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
import cv2
from tqdm import tqdm

from env.wrappers import make_env
from algorithms.hifno import HiFNOEncoder, HiFNOAgent
from algorithms.models.HiFNO import HierarchicalFNO
import utils


VIDEO_FRAMES = 100  # 每个片段的帧数
FPS = 30  # 帧率
FRAME_INDICES = [0, 200, 600, 800]  
FRAME_SKIP = 5  # 每隔多少帧采样一次
ACTION_NOISE = 0.3  # 噪声


SEGMENT_SEEDS = [42, 100, 200, 300]  

ENV_DOMAIN = "walker"
ENV_TASK = "walk"


MODEL_NAME = "svea_walker_walk_20241224_111447"
MODEL_STEP = "500000"
MODEL_PATH = "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/logs/svea/svea_walker_walk_20241224_111447/model/500000.pt"

class Args:
    def __init__(self):
        self.frame_stack = 3
        self.hidden_dim = 256
        self.embed_dim = 128
        self.depth = 2
        self.num_scales = 3

def create_env(domain_name=ENV_DOMAIN, task_name=ENV_TASK, seed=42):
    env = make_env(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
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
    if not isinstance(obs, np.ndarray) and not isinstance(obs, torch.Tensor):
        obs = np.array(obs)
        
    if not isinstance(obs, torch.Tensor):
        obs = torch.FloatTensor(obs)
    if len(obs.shape) == 3:
        obs = obs.unsqueeze(0)
    
    obs = obs.to(encoder.device)
    
    features = []
    
    def hook_fn(module, input, output):
        features.append(output.detach())
    
    hooks = []
    for scale_idx in range(encoder.hifno.num_scales):
        last_conv = encoder.hifno.conv_res_fourier_layers[scale_idx][-1]
        hook = last_conv.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    with torch.no_grad():
        _ = encoder.hifno(obs)
    
    for hook in hooks:
        hook.remove()
    
    return features

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_total_frames(env):
    episode_length = env._max_episode_steps 
    action_repeat = env.action_repeat if hasattr(env, 'action_repeat') else 1 
    total_frames = episode_length // action_repeat 
    return total_frames

def load_agent(model_path):
    print(f"加载模型: {model_path}")
    agent = torch.load(model_path)
    agent.eval()  
    return agent

def fig_to_image(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img

def create_frame_visualization(obs, encoder=None):
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    
    num_stacked_frames = obs.shape[0] // 3
    
    # 计算需要的列数
    cols_needed = 3 + 3  # 每个时间步的3个通道 + 3个特征尺度
    

    fig = plt.figure(figsize=(max(16, cols_needed * 2.5), 9))
    

    latest_frame = obs[-3:]  # 获取最后3个通道
    obs_tensor = torch.FloatTensor(latest_frame)
    

    fft_magnitudes = []
    for c in range(3):  
        magnitude = compute_fft_2d(obs_tensor[c])
        fft_magnitudes.append(magnitude)
    

    for i in range(3):
        ax1 = plt.subplot2grid((2, cols_needed), (0, i), colspan=1)
        ax1.imshow(latest_frame[i], cmap='plasma')
        ax1.set_title(f'Channel {i}')
        ax1.axis('off')
        
        # FFT
        ax2 = plt.subplot2grid((2, cols_needed), (1, i), colspan=1)
        ax2.imshow(torch.log(fft_magnitudes[i] + 1).numpy(), cmap='viridis')
        ax2.set_title(f'Channel {i} FFT')
        ax2.axis('off')
    
    
    if encoder is not None:
        
        features = visualize_fno_features(encoder, obs)
        
        for scale_idx, scale_features in enumerate(features):
            
            fft_magnitude = compute_fft_2d(scale_features[0])
            avg_magnitude = torch.mean(fft_magnitude, dim=0)
            
            
            ax3 = plt.subplot2grid((2, cols_needed), (0, 3+scale_idx), colspan=1)
            ax3.imshow(torch.log(avg_magnitude + 1).cpu(), cmap='plasma')
            ax3.set_title(f'Scale {scale_idx} Avg FFT')
            ax3.axis('off')
            
            
            ax4 = plt.subplot2grid((2, cols_needed), (1, 3+scale_idx), colspan=1)
            channel_idx = 0  
            ax4.imshow(scale_features[0, channel_idx].cpu(), cmap='inferno')
            ax4.set_title(f'Scale {scale_idx} Feature')
            ax4.axis('off')
    
    plt.tight_layout()
    
    
    img = fig_to_image(fig)
    plt.close(fig)
    
    return img

def collect_frames_with_agent(env, frame_indices, agent):
    obs = env.reset()
    observations = []
    
    max_frame = max(frame_indices)
    for i in range(max_frame + 1):
        if i in frame_indices:
            observations.append(np.array(obs))
        
        with torch.no_grad():
            action = agent.select_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    return observations

def create_multi_segment_video(agent, encoder, frame_indices, frames_per_segment, output_path, fps=30, segment_seeds=SEGMENT_SEEDS):
    print(f"开始创建多段视频，起始帧: {frame_indices}, 每段帧数: {frames_per_segment}")
    
    
    if len(segment_seeds) < len(frame_indices):
        segment_seeds = segment_seeds + [s+1000 for s in segment_seeds]
        segment_seeds = segment_seeds[:len(frame_indices)]
    
    
    observations = []
    for i, (frame_idx, seed) in enumerate(zip(frame_indices, segment_seeds[:len(frame_indices)])):
       
        env = create_env(seed=seed)
        print(f"片段 {i} 创建环境，使用种子 {seed}")
        
        
        obs = env.reset()
        for j in range(frame_idx):
            with torch.no_grad():
                action = agent.select_action(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
        
        observations.append((np.array(obs), env))  
    
    
    first_frame = create_frame_visualization(observations[0][0], encoder)
    height, width, layers = first_frame.shape
    
   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    
    for segment_idx, (start_obs, env) in enumerate(observations):
        print(f"处理片段 {segment_idx}, 起始帧 {frame_indices[segment_idx]}, 种子 {segment_seeds[segment_idx]}")
        
        obs = start_obs
        
        frame = create_frame_visualization(obs, encoder)
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        for i in tqdm(range(1, frames_per_segment), desc=f"片段 {segment_idx}"):
            with torch.no_grad():
                action = agent.select_action(obs)
            
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
            
            frame = create_frame_visualization(obs, encoder)
            
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    video.release()
    print(f"视频已保存到: {output_path}")

def main():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(current_dir, 'visualization_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出目录: {output_dir}")
    
    model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件: {model_path}")
        return
        
    agent = load_agent(model_path)
    print("模型加载成功")
    
    timestamp = get_timestamp()
    args = Args()
    
    env = create_env()  
    
    total_frames = get_total_frames(env)
    print(f"环境设置:")
    print(f"- 回合长度: {env._max_episode_steps}")
    print(f"- 动作重复: {env.action_repeat if hasattr(env, 'action_repeat') else 1}")
    print(f"- 可用总帧数: {total_frames}")
    
    if max(FRAME_INDICES) >= total_frames:
        print(f"警告: 部分起始帧索引超出了总帧数!")
        print(f"请选择0到{total_frames-1}之间的帧索引")
        return
    
    obs = env.reset()

    obs_array = np.array(obs)
    print(f"观察形状: {obs_array.shape}")
    
    encoder = HiFNOEncoder(
        obs_shape=obs_array.shape,
        feature_dim=args.embed_dim,
        args=args
    )
    

    frame_indices_str = '_'.join(map(str, FRAME_INDICES))
    video_path = os.path.join(output_dir, f'freq_domain_video_{MODEL_NAME}_frames{frame_indices_str}_{timestamp}.mp4')
    

    create_multi_segment_video(
        agent=agent,
        encoder=encoder,
        frame_indices=FRAME_INDICES,
        frames_per_segment=VIDEO_FRAMES,
        output_path=video_path,
        fps=FPS,
        segment_seeds=SEGMENT_SEEDS
    )

if __name__ == "__main__":
    main()



'''
python /mnt/lustre/GPU4/home/wuhanpeng/dmcontrol/src/vis_freq_domain_video.py

'''