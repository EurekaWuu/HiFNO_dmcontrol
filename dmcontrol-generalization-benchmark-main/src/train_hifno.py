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


def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
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
	utils.write_info(args, os.path.join(work_dir, 'info.log'))
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None)

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	torch.cuda.empty_cache()  # 清理GPU缓存
	
	if args.algorithm == 'hifno_bisim':
		replay_buffer = utils.BisimReplayBuffer(
			obs_shape=env.observation_space.shape,
			action_shape=env.action_space.shape,
			capacity=args.train_steps,
			batch_size=args.batch_size * 2
		)
	elif args.algorithm == 'hifno_bisim_1':
		# hifno_bisim_1设置较小的batch_size
		os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
		replay_buffer = utils.BisimReplayBuffer(
			obs_shape=env.observation_space.shape,
			action_shape=env.action_space.shape,
			capacity=args.train_steps,
			batch_size=16
		)
	else:
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
				update_info = agent.update(replay_buffer, L, step)
				if args.algorithm == 'hifno_bisim' and L is not None:
					L.log('train/bisim_loss', update_info.get('bisim_loss', 0), step)
					L.log('train/L_BB', update_info.get('L_BB', 0), step)
					L.log('train/L_ICC', update_info.get('L_ICC', 0), step)
					L.log('train/L_CC', update_info.get('L_CC', 0), step)
				elif args.algorithm == 'hifno_bisim_1' and L is not None:
					L.log('train/clip_loss', update_info.get('clip_loss', 0), step)
					L.log('train/sc_loss', update_info.get('sc_loss', 0), step)
					L.log('train/clip_bisim_loss', update_info.get('clip_bisim_loss', 0), step)
					L.log('train/total_loss', update_info.get('total_loss', 0), step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	main(args)


'''
CUDA_VISIBLE_DEVICES=3 python train_hifno.py --algorithm hifno --hidden_dim 128 --domain_name walker --task_name walk --seed 1 --lr 1e-4 --embed_dim 256 --batch_size 32
CUDA_VISIBLE_DEVICES=6 python train_hifno.py --algorithm hifno_bisim --hidden_dim 128 --domain_name walker --task_name walk --seed 1 --lr 1e-4 --embed_dim 256 --batch_size 32
CUDA_VISIBLE_DEVICES=7 python train_hifno.py --algorithm hifno_bisim_1 --hidden_dim 128 --domain_name walker --task_name walk --seed 1 --lr 1e-4 --embed_dim 128 --batch_size 24 --lambda_SC 0.5 --lambda_clip 0.2 --clip_loss_weight 0.3

# 只使用语义类内一致性损失
CUDA_VISIBLE_DEVICES=6 python train_hifno.py --algorithm hifno_bisim_1 --hidden_dim 128 --domain_name walker --task_name walk --seed 1 --lr 1e-4 --embed_dim 128 --batch_size 24 --use_sc_loss True --use_clip_bisim_loss False --lambda_SC 1 --clip_loss_weight 0.4

# 只使用CLIP引导的双模拟损失
CUDA_VISIBLE_DEVICES=4 python train_hifno.py --algorithm hifno_bisim_1 --hidden_dim 128 --domain_name walker --task_name walk --seed 1 --lr 1e-4 --embed_dim 128 --batch_size 24 --use_sc_loss False --use_clip_bisim_loss True --lambda_clip 1 --clip_loss_weight 0.4

# 同时使用两种损失
CUDA_VISIBLE_DEVICES=4 python train_hifno.py --algorithm hifno_bisim_1 --use_sc_loss True --use_clip_bisim_loss True --lambda_SC 0.5 --lambda_clip 0.5 --batch_size 16
'''