import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,2,4'

import torch
import torch.nn as nn
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory_multigpu import make_agent
from logger import Logger
from video import VideoRecorder
from datetime import datetime
from algorithms.hifno_multigpu import HiFNOAgent
from torch import nn


def safe_sample_action(agent, obs):
	
	if isinstance(agent, nn.DataParallel):
		return agent.module.sample_action(obs)
	else:
		# 如果不是 DataParallel，先包装成 DataParallel
		agent = nn.DataParallel(agent)
		return agent.module.sample_action(obs)

def safe_select_action(agent, obs):
	
	if isinstance(agent, nn.DataParallel):
		return agent.module.select_action(obs)
	else:
		# 如果不是 DataParallel，先包装成 DataParallel
		agent = nn.DataParallel(agent)
		return agent.module.select_action(obs)


def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				if not isinstance(obs, np.ndarray):
					obs = np.array(obs)
				
				if len(obs.shape) == 3:
					obs = obs.reshape(1, *obs.shape)
				elif len(obs.shape) == 2:
					obs = obs.reshape(1, 1, *obs.shape)
				
				obs = torch.from_numpy(obs).float().cuda()

				action = agent.module.select_action(obs)
				obs, reward, done, _ = env.step(action)
				video.record(env)
				episode_reward += reward

		if L is not None:
			_test_env = '_test_env' if test_env else ''
			video.save(f'{step}{_test_env}.mp4')
			L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
		episode_rewards.append(episode_reward)
	
	return np.mean(episode_rewards)


def main(args):
	
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
	utils.write_info(args, os.path.join(work_dir, 'info.log'))
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None)

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	print(f"Number of available GPUs: {torch.cuda.device_count()}")
	
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
	print(f"Agent type: {type(agent).__module__}.{type(agent).__name__}")

	
	device = torch.device('cuda')
	agent = agent.to(device)

	# 无论单卡还是多卡都包装成 DataParallel
	print(f"Using {torch.cuda.device_count()} GPU(s)!")
	agent = nn.DataParallel(agent)

	start_step, episode, episode_reward, done = 0, 0, 0, True
	L = Logger(work_dir)
	start_time = time.time()
	
	# Training loop
	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				evaluate(env, agent, video, args.eval_episodes, L, step)
				if test_env is not None:
					evaluate(test_env, agent, video, args.eval_episodes, L, step, test_env=True)
				L.dump(step)

			# Save agent periodically
			if step > start_step and step % args.save_freq == 0:
				
				if isinstance(agent.encoder, torch.nn.DataParallel):
					encoder_state = agent.encoder.module.state_dict()
					actor_state = agent.actor.module.state_dict()
					critic_state = agent.critic.module.state_dict()
				else:
					encoder_state = agent.encoder.state_dict()
					actor_state = agent.actor.state_dict() 
					critic_state = agent.critic.state_dict()

				torch.save(
					{
						'encoder': encoder_state,
						'actor': actor_state,
						'critic': critic_state
					},
					os.path.join(model_dir, f'{step}.pt')
				)

			L.log('train/episode_reward', episode_reward, step)

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step)

		# Sample action for data collection
		if step < args.init_steps:
			# 处理 LazyFrames
			if not isinstance(obs, np.ndarray):
				obs = np.array(obs)
			
			if len(obs.shape) == 3:  # (C, H, W)
				obs = obs.reshape(1, *obs.shape)
			elif len(obs.shape) == 2:  # (H, W)
				obs = obs.reshape(1, 1, *obs.shape)
			
			obs_tensor = torch.FloatTensor(obs).cuda()
			action = agent.module.sample_action(obs_tensor)
		else:
			with utils.eval_mode(agent):
				# 处理 LazyFrames
				if not isinstance(obs, np.ndarray):
					obs = np.array(obs)
				
				if len(obs.shape) == 3:  # (C, H, W)
					obs = obs.reshape(1, *obs.shape)
				elif len(obs.shape) == 2:  # (H, W)
					obs = obs.reshape(1, 1, *obs.shape)
				
				obs_tensor = torch.FloatTensor(obs).cuda()
				action = agent.module.sample_action(obs_tensor)

		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			for _ in range(num_updates):
				agent.module.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		
		# 处理 LazyFrames 用于存储
		if not isinstance(obs, np.ndarray):
			obs_array = np.array(obs)
		else:
			obs_array = obs
			
		if not isinstance(next_obs, np.ndarray):
			next_obs_array = np.array(next_obs)
		else:
			next_obs_array = next_obs
			
		replay_buffer.add(obs_array, action, reward, next_obs_array, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	main(args)


# python train_hifno_multigpu.py --algorithm hifno_multigpu --hidden_dim 128 --domain_name walker --task_name walk --seed 1 --lr 1e-4 --embed_dim 256 --batch_size 32