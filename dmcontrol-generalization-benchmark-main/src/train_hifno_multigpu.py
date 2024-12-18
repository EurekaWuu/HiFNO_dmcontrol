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
from algorithms.hifno_multigpu import HiFNOAgent
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'


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

	# 将agent移到GPU并启用多GPU，设置为DataParallel
	if torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs!")
		
		# 修改这里：先将模型移到cuda
		agent.encoder = agent.encoder.cuda()
		agent.actor = agent.actor.cuda()
		agent.critic = agent.critic.cuda()
		agent.critic_target = agent.critic_target.cuda()
		
		# 然后包装为DataParallel
		agent.encoder = torch.nn.DataParallel(
			agent.encoder,
			device_ids=list(range(torch.cuda.device_count()))
		)
		
		agent.actor = torch.nn.DataParallel(
			agent.actor,
			device_ids=list(range(torch.cuda.device_count()))
		)
		
		agent.critic = torch.nn.DataParallel(
			agent.critic,
			device_ids=list(range(torch.cuda.device_count()))
		)
		
		agent.critic_target = torch.nn.DataParallel(
			agent.critic_target,
			device_ids=list(range(torch.cuda.device_count()))
		)

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
				# 保存模型时需要特别处理DataParallel
				if isinstance(agent.encoder, torch.nn.DataParallel):
					encoder_state = agent.encoder.module.state_dict()
					actor_state = agent.actor.module.state_dict()
					critic_state = agent.critic.module.state_dict()
					critic_target_state = agent.critic_target.module.state_dict()
				else:
					encoder_state = agent.encoder.state_dict()
					actor_state = agent.actor.state_dict() 
					critic_state = agent.critic.state_dict()
					critic_target_state = agent.critic_target.state_dict()

				torch.save(
					{
						'encoder': encoder_state,
						'actor': actor_state,
						'critic': critic_state,
						'critic_target': critic_target_state
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

	print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	main(args)


#CUDA_VISIBLE_DEVICES=3 python train.py --algorithm hifno --hidden_dim 128 --domain_name walker --task_name walk --seed 1 --lr 1e-4 --embed_dim 256 --batch_size 32