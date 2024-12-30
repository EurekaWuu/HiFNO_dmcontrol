import os
import sys
import numpy as np
import torch
import gym
from datetime import datetime
import matplotlib.pyplot as plt

from env.wrappers import make_env
from algorithms.hifno import HiFNOEncoder, HiFNOAgent
import utils

ORIGINAL_ENV_DOMAIN = "walker"
ORIGINAL_ENV_TASK = "walk"
MODEL_PATH = "/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol-generalization-benchmark-main/dmcontrol-generalization-benchmark-main/src/logs/walker_walk/hifno/1/20241209_150036/model/500000.pt"

TEST_ENVS = [
    ("cheetah", "run"),
    ("walker", "stand"),
    ("walker", "run"),
    ("quadruped", "walk"),
    ("humanoid", "stand"),
    ("humanoid_CMU", "stand"),
    ("hopper", "stand"),
]

NUM_TEST_EPISODES = 30
EPISODE_LENGTH = 1000
SEED = 1

class Args:
    def __init__(self):
        self.frame_stack = 3
        self.hidden_dim = 256
        self.embed_dim = 128
        self.depth = 2
        self.num_scales = 3

def load_agent(model_path):
    print(f"Loading model from: {model_path}")
    agent = torch.load(model_path)
    agent.eval()
    return agent

def evaluate_env(env, agent, num_episodes=5):
    episode_rewards = []
    episode_lengths = []
    success_rate = 0
    avg_action_magnitude = []
    early_termination = 0
    reward_distributions = []
    action_smoothness = []
    task_specific_metrics = {
        'max_height': [],
        'forward_velocity': [],
        'energy_efficiency': [],
        'balance_metric': []
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        episode_rewards_step = []
        
        prev_action = None
        while not done:
            with torch.no_grad():
                action = agent.select_action(obs)
                episode_actions.append(action)
                
                if prev_action is not None:
                    action_smoothness.append(np.linalg.norm(action - prev_action))
                prev_action = action
                
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            episode_rewards_step.append(reward)
            
            if 'height' in info:
                task_specific_metrics['max_height'].append(info['height'])
            if 'forward_velocity' in info:
                task_specific_metrics['forward_velocity'].append(info['forward_velocity'])
            if 'energy' in info:
                task_specific_metrics['energy_efficiency'].append(reward / (np.linalg.norm(action) + 1e-8))
            if 'balance' in info:
                task_specific_metrics['balance_metric'].append(info['balance'])
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        avg_action_magnitude.append(np.mean([np.linalg.norm(a) for a in episode_actions]))
        
        reward_distributions.append({
            'mean': np.mean(episode_rewards_step),
            'std': np.std(episode_rewards_step),
            'min': np.min(episode_rewards_step),
            'max': np.max(episode_rewards_step),
            'quartiles': np.percentile(episode_rewards_step, [25, 50, 75])
        })
        
        if episode_reward > env._max_episode_steps * 0.8:
            success_rate += 1
        
        if episode_length < env._max_episode_steps:
            early_termination += 1
    
    success_rate = success_rate / num_episodes
    early_termination_rate = early_termination / num_episodes
    mean_action_smoothness = np.mean(action_smoothness) if action_smoothness else 0
    
    task_metrics = {}
    for metric_name, values in task_specific_metrics.items():
        if values:
            task_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values)
            }
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'all_rewards': episode_rewards,
        'success_rate': success_rate,
        'early_termination_rate': early_termination_rate,
        'mean_action_magnitude': np.mean(avg_action_magnitude),
        'std_action_magnitude': np.std(avg_action_magnitude),
        'reward_distributions': reward_distributions,
        'action_smoothness': mean_action_smoothness,
        'task_specific_metrics': task_metrics
    }

def get_action_dim(domain, task):
    env = make_env(domain_name=domain, task_name=task)
    return env.action_space.shape[0]

def visualize_generalization_results(results_dict, output_dir):
    plt.style.use('seaborn')
    plt.rcParams['figure.dpi'] = 100

    COLORS = {
        'original': '#0072BD',
        'env1': '#D95319',
        'env2': '#EDB120',
        'env3': '#7E2F8E',
        'env4': '#77AC30',
        'env5': '#4DBEEE',
        'env6': '#A2142F',
    }

    env_names = list(results_dict.keys())
    env_colors = {}
    available_colors = [color for key, color in COLORS.items() if key != 'original']
    
    for i, env_name in enumerate(env_names):
        if 'Original' in env_name:
            env_colors[env_name] = COLORS['original']
        else:
            color_index = (i - 1) % len(available_colors)
            env_colors[env_name] = available_colors[color_index]

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    
    x = np.arange(len(env_names))
    
    performance_grid = gs[0:2, :].subgridspec(2, 2)
    
    ax1 = fig.add_subplot(performance_grid[0, 0])
    mean_rewards = [results_dict[env]['mean_reward'] for env in env_names]
    std_rewards = [results_dict[env]['std_reward'] for env in env_names]
    ax1.bar(x, mean_rewards, yerr=std_rewards, capsize=5,
            color=[env_colors[env] for env in env_names])
    ax1.set_xticks(x)
    ax1.set_xticklabels(env_names, rotation=45, ha='right')
    ax1.set_ylabel('Mean Reward ± Std')
    ax1.set_title('Mean Rewards Comparison')
    
    ax2 = fig.add_subplot(performance_grid[0, 1])
    success_rates = [results_dict[env]['success_rate'] * 100 for env in env_names]
    ax2.bar(x, success_rates,
            color=[env_colors[env] for env in env_names])
    ax2.set_xticks(x)
    ax2.set_xticklabels(env_names, rotation=45, ha='right')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rates Comparison')
    
    ax3 = fig.add_subplot(performance_grid[1, 0])
    smoothness = [results_dict[env]['action_smoothness'] for env in env_names]
    ax3.bar(x, smoothness,
            color=[env_colors[env] for env in env_names])
    ax3.set_xticks(x)
    ax3.set_xticklabels(env_names, rotation=45, ha='right')
    ax3.set_ylabel('Action Smoothness')
    ax3.set_title('Action Smoothness Comparison')
    
    ax4 = fig.add_subplot(performance_grid[1, 1])
    action_magnitudes = [results_dict[env]['mean_action_magnitude'] for env in env_names]
    action_magnitude_stds = [results_dict[env]['std_action_magnitude'] for env in env_names]
    ax4.bar(x, action_magnitudes, yerr=action_magnitude_stds, capsize=5,
            color=[env_colors[env] for env in env_names])
    ax4.set_xticks(x)
    ax4.set_xticklabels(env_names, rotation=45, ha='right')
    ax4.set_ylabel('Action Magnitude ± Std')
    ax4.set_title('Action Magnitude Comparison')
    
    ax5 = fig.add_subplot(gs[2, 0])
    reward_data = []
    labels = []
    colors = []
    for env_name in env_names:
        all_episode_rewards = []
        for dist in results_dict[env_name]['reward_distributions']:
            all_episode_rewards.extend([dist['mean']])
        reward_data.append(all_episode_rewards)
        labels.append(env_name)
        colors.append(env_colors[env_name])
    
    bp = ax5.boxplot(reward_data, labels=labels, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=colors[i], alpha=0.7)
    ax5.set_xticklabels(labels, rotation=45, ha='right')
    ax5.set_ylabel('Reward Distribution')
    ax5.set_title('Reward Distributions Across Environments')
    
    ax6 = fig.add_subplot(gs[2, 1])
    for env_name in env_names:
        rewards = results_dict[env_name]['all_rewards']
        episodes = range(len(rewards))
        ax6.plot(episodes, rewards, 
                label=env_name, 
                marker='o',
                color=env_colors[env_name])
    
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Total Reward')
    ax6.set_title('Reward Trends Across Episodes')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generalization_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_dir = '/mnt/lustre/GPU4/home/wuhanpeng/dmcontrol-generalization-benchmark-main/dmcontrol-generalization-benchmark-main'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, 'generalization_results', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory at: {output_dir}")
    
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at: {model_path}")
        return
    
    agent = load_agent(model_path)
    print("Model loaded successfully")
    
    results_file = os.path.join(output_dir, 'generalization_results.txt')
    
    original_dim = get_action_dim(ORIGINAL_ENV_DOMAIN, ORIGINAL_ENV_TASK)
    print(f"Original environment action dimension: {original_dim}")
    
    compatible_envs = []
    for domain, task in TEST_ENVS:
        test_dim = get_action_dim(domain, task)
        if test_dim == original_dim:
            compatible_envs.append((domain, task))
            print(f"Environment {domain}-{task} is compatible (dim={test_dim})")
        else:
            print(f"Skipping {domain}-{task} due to incompatible action dimension (dim={test_dim})")
    
    print(f"\nTesting on original environment: {ORIGINAL_ENV_DOMAIN}-{ORIGINAL_ENV_TASK}")
    original_env = make_env(
        domain_name=ORIGINAL_ENV_DOMAIN,
        task_name=ORIGINAL_ENV_TASK,
        seed=42,
        episode_length=1000,
        action_repeat=1,
        image_size=84,
        mode='train'
    )
    original_results = evaluate_env(original_env, agent, NUM_TEST_EPISODES)
    
    with open(results_file, 'w') as f:
        f.write(f"Original Environment ({ORIGINAL_ENV_DOMAIN}-{ORIGINAL_ENV_TASK}):\n")
        f.write(f"Action dimension: {original_dim}\n")
        f.write(f"Mean Reward: {original_results['mean_reward']:.2f} ± {original_results['std_reward']:.2f}\n")
        f.write(f"Mean Episode Length: {original_results['mean_length']:.2f} ± {original_results['std_length']:.2f}\n")
        f.write(f"Success Rate: {original_results['success_rate']:.2%}\n")
        f.write(f"Early Termination Rate: {original_results['early_termination_rate']:.2%}\n")
        f.write(f"Mean Action Magnitude: {original_results['mean_action_magnitude']:.4f} ± {original_results['std_action_magnitude']:.4f}\n")
        f.write(f"Action Smoothness: {original_results['action_smoothness']:.4f}\n")
        
        f.write("\nReward Distribution Statistics:\n")
        for i, dist in enumerate(original_results['reward_distributions']):
            f.write(f"Episode {i+1}:\n")
            f.write(f"  Mean: {dist['mean']:.2f}, Std: {dist['std']:.2f}\n")
            f.write(f"  Min: {dist['min']:.2f}, Max: {dist['max']:.2f}\n")
            f.write(f"  Quartiles (25, 50, 75): {dist['quartiles']}\n")
        
        if original_results['task_specific_metrics']:
            f.write("\nTask Specific Metrics:\n")
            for metric_name, values in original_results['task_specific_metrics'].items():
                f.write(f"{metric_name}:\n")
                f.write(f"  Mean: {values['mean']:.4f} ± {values['std']:.4f}\n")
                f.write(f"  Range: [{values['min']:.4f}, {values['max']:.4f}]\n")
        
        f.write(f"\nAll Rewards: {original_results['all_rewards']}\n\n")
    
    all_results = {
        f"Original Environment ({ORIGINAL_ENV_DOMAIN}-{ORIGINAL_ENV_TASK})": original_results
    }
    
    for domain, task in compatible_envs:
        print(f"\nTesting on compatible environment: {domain}-{task}")
        env = make_env(
            domain_name=domain,
            task_name=task,
            seed=42,
            episode_length=1000,
            action_repeat=1,
            image_size=84,
            mode='train'
        )
        
        results = evaluate_env(env, agent, NUM_TEST_EPISODES)
        
        with open(results_file, 'a') as f:
            f.write("="*100 + "\n")  
            f.write(f"Test Environment ({domain}-{task}):\n")
            f.write(f"Action dimension: {get_action_dim(domain, task)}\n")
            f.write(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
            f.write(f"Mean Episode Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}\n")
            f.write(f"Success Rate: {results['success_rate']:.2%}\n")
            f.write(f"Early Termination Rate: {results['early_termination_rate']:.2%}\n")
            f.write(f"Mean Action Magnitude: {results['mean_action_magnitude']:.4f} ± {results['std_action_magnitude']:.4f}\n")
            f.write(f"Action Smoothness: {results['action_smoothness']:.4f}\n")
            
            f.write("\nReward Distribution Statistics:\n")
            for i, dist in enumerate(results['reward_distributions']):
                f.write(f"Episode {i+1}:\n")
                f.write(f"  Mean: {dist['mean']:.2f}, Std: {dist['std']:.2f}\n")
                f.write(f"  Min: {dist['min']:.2f}, Max: {dist['max']:.2f}\n")
                f.write(f"  Quartiles (25, 50, 75): {dist['quartiles']}\n")
            
            if results['task_specific_metrics']:
                f.write("\nTask Specific Metrics:\n")
                for metric_name, values in results['task_specific_metrics'].items():
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Mean: {values['mean']:.4f} ± {values['std']:.4f}\n")
                    f.write(f"  Range: [{values['min']:.4f}, {values['max']:.4f}]\n")
            
            f.write(f"\nAll Rewards: {results['all_rewards']}\n\n")
        
        all_results[f"Test Environment ({domain}-{task})"] = results
        print(f"Results saved for {domain}-{task}")
    
    visualize_generalization_results(all_results, output_dir)
    print(f"Visualization saved to {output_dir}/generalization_performance.png")

if __name__ == "__main__":
    main() 



'''
python test_generalization.py

'''