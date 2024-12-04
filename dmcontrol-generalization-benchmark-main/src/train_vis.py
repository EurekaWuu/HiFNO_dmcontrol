import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

class VisualLogger:
    def __init__(self, work_dir):
        self.writer = SummaryWriter(work_dir)
        self.episode_rewards = []
        self.eval_rewards = []
        
    def log_training_metrics(self, episode_reward, step, duration):
        self.episode_rewards.append(episode_reward)
        
        # 记录基础指标
        self.writer.add_scalar('Training/Episode_Reward', episode_reward, step)
        self.writer.add_scalar('Training/Duration', duration, step)
        
        # 每100步绘制奖励分布
        if len(self.episode_rewards) % 100 == 0:
            plt.figure(figsize=(10, 5))
            sns.histplot(data=self.episode_rewards[-100:])
            plt.title('Recent 100 Episodes Reward Distribution')
            self.writer.add_figure('Training/Reward_Distribution', plt.gcf(), step)
            plt.close()
    
    def log_evaluation_metrics(self, eval_reward, step, test_env=False):
        env_type = 'Test' if test_env else 'Eval'
        self.writer.add_scalar(f'{env_type}/Episode_Reward', eval_reward, step)
        self.eval_rewards.append(eval_reward)

def evaluate(env, agent, video, num_episodes, L, visual_logger, step, test_env=False):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i==0))
        done = False
        episode_reward = 0
        
        # 收集每个episode的状态和动作
        states = []
        actions = []
        
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            states.append(obs)
            actions.append(action)
            
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        if L is not None:
            _test_env = '_test_env' if test_env else ''
            video.save(f'{step}{_test_env}.mp4')
            L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
            
        episode_rewards.append(episode_reward)
        
        # 可视化状态和动作分布
        if i == 0:  # 只为第一个episode创建可视化
            states = np.array(states)
            actions = np.array(actions)
            
            # 状态空间可视化
            plt.figure(figsize=(10, 5))
            for j in range(min(3, states.shape[1])):  # 只显示前3个维度
                plt.plot(states[:, j], label=f'Dim {j}')
            plt.title('State Space Trajectory')
            plt.legend()
            visual_logger.writer.add_figure(f'Evaluation/State_Trajectory', plt.gcf(), step)
            plt.close()
            
            # 动作空间可视化
            plt.figure(figsize=(10, 5))
            for j in range(actions.shape[1]):
                plt.plot(actions[:, j], label=f'Action {j}')
            plt.title('Action Trajectory')
            plt.legend()
            visual_logger.writer.add_figure(f'Evaluation/Action_Trajectory', plt.gcf(), step)
            plt.close()
    
    mean_reward = np.mean(episode_rewards)
    visual_logger.log_evaluation_metrics(mean_reward, step, test_env)
    return mean_reward

def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode='train'
    )
    test_env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed+42,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.eval_mode,
        intensity=args.distracting_cs_intensity
    ) if args.eval_mode is not None else None

    # Create working directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, 
                           args.algorithm, str(args.seed), timestamp)
    print('Working directory:', work_dir)
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None)

    # 初始化可视化工具
    visual_logger = VisualLogger(work_dir)

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size
    )
    cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
    print('Observations:', env.observation_space.shape)
    print('Cropped observations:', cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )

    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir)
    start_time = time.time()
    
    for step in range(start_step, args.train_steps+1):
        if done:
            if step > start_step:
                duration = time.time() - start_time
                visual_logger.log_training_metrics(episode_reward, step, duration)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                print('Evaluating:', work_dir)
                L.log('eval/episode', episode, step)
                evaluate(env, agent, video, args.eval_episodes, L, visual_logger, step)
                if test_env is not None:
                    evaluate(test_env, agent, video, args.eval_episodes, L, visual_logger, step, test_env=True)
                L.dump(step)

            # Save agent periodically
            if step > start_step and step % args.save_freq == 0:
                torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

        # 添加环境渲染
        if step % args.render_freq == 0:  # 每render_freq步渲染一次
            frame = env.render(mode='rgb_array')
            visual_logger.writer.add_image('Environment/render', 
                                         torch.from_numpy(frame).permute(2,0,1), 
                                         step)

    print('Completed training for', work_dir)

if __name__ == '__main__':
    args = parse_args()
    # 添加渲染频率参数
    args.render_freq = 1000  # 默认每1000步渲染一次
    main(args) 