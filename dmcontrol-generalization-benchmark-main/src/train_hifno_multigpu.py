import os

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def process_obs(obs, device):
	"""统一处理观测数据的维度转换"""
	if not isinstance(obs, np.ndarray):
		obs = np.array(obs)
	
	# 检查是 float32 类型
	if obs.dtype != np.float32:
		obs = obs.astype(np.float32)
	
	if len(obs.shape) == 3:  # (C, H, W)
		obs = np.expand_dims(obs, axis=0)  # 添加 batch 维度 (1, C, H, W)
	elif len(obs.shape) == 2:  # (H, W)
		obs = np.expand_dims(np.expand_dims(obs, axis=0), axis=0)  # (1, 1, H, W)
		
	return torch.FloatTensor(obs).to(device)


def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				device = next(agent.parameters()).device
				obs_tensor = process_obs(obs, device)
				action = agent.module.select_action(obs_tensor)
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
	# 初始化进程组,获取 rank 和 world_size
	dist.init_process_group(backend='nccl')
	local_rank = int(os.environ.get("LOCAL_RANK", 0))
	
	torch.cuda.set_device(local_rank)
	device = torch.device("cuda", local_rank)

	utils.set_seed_everywhere(args.seed + local_rank)

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
	if dist.get_rank() == 0:
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		work_dir = os.path.join(
			args.log_dir, 
			args.domain_name + '_' + args.task_name,
			args.algorithm, 
			str(args.seed), 
			timestamp
		)
		print('Working directory:', work_dir)
		utils.make_dir(work_dir)
		utils.write_info(args, os.path.join(work_dir, 'info.log'))
		model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
		video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
		video = VideoRecorder(video_dir if args.save_video else None)
		L = Logger(work_dir)
	dist.barrier()

	
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

	# 包装成 DDP
	agent = agent.to(device)
	agent = DDP(agent, device_ids=[local_rank], output_device=local_rank)

	start_step, episode, episode_reward, done = 0, 0, 0, True
	start_time = time.time()
	
	# Training loop
	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				# 仅在主进程做日志
				if dist.get_rank() == 0:
					L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				# 仅在主进程dump
				if dist.get_rank() == 0:
					L.dump(step)

			# Evaluate agent periodically (仅在主进程做评测和日志)
			if dist.get_rank() == 0 and step % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				
				evaluate(env, agent, video, args.eval_episodes, L, step)
				if test_env is not None:
					evaluate(test_env, agent, video, args.eval_episodes, L, step, test_env=True)
				L.dump(step)

			# Save agent periodically(只在 rank=0 进行)
			if dist.get_rank() == 0 and step > start_step and step % args.save_freq == 0:
				torch.save(
					{
						'encoder': agent.module.encoder.state_dict(),
						'actor': agent.module.actor.state_dict(),
						'critic': agent.module.critic.state_dict()
					},
					os.path.join(model_dir, f'{step}.pt')
				)

			if dist.get_rank() == 0:
				L.log('train/episode_reward', episode_reward, step)

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			if dist.get_rank() == 0:
				L.log('train/episode', episode, step)

		# Sample action for data collection
		if step < args.init_steps:
			obs_tensor = process_obs(obs, device)
			action = agent.module.sample_action(obs_tensor)
		else:
			with utils.eval_mode(agent):
				obs_tensor = process_obs(obs, device)
				action = agent.module.sample_action(obs_tensor)
		
		# Update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			for _ in range(num_updates):
				# 只在 rank 0 进程传入 Logger
				agent.module.update(replay_buffer, L if dist.get_rank() == 0 else None, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		
		# 处理用于存储的观测数据
		obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
		next_obs_array = np.array(next_obs) if not isinstance(next_obs, np.ndarray) else next_obs
		
		
		obs_array = obs_array.astype(np.float32)
		next_obs_array = next_obs_array.astype(np.float32)
		
		replay_buffer.add(obs_array, action, reward, next_obs_array, done_bool)
		episode_reward += reward
		
		episode_step += 1

	if dist.get_rank() == 0:
		print('Completed training for', work_dir)
	dist.destroy_process_group()


if __name__ == '__main__':
	args = parse_args()
	main(args)


# python train_hifno_multigpu.py --algorithm hifno_multigpu --hidden_dim 128 --domain_name walker --task_name walk --seed 1 --lr 1e-4 --embed_dim 256 --batch_size 32