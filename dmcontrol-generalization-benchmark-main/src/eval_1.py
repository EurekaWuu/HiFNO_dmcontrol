import torch
import torchvision
import os
import numpy as np
import gym
import utils
from copy import deepcopy
from tqdm import tqdm
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import augmentations
from pathlib import Path


def evaluate(env, agent, video, num_episodes, eval_mode, L, step, adapt=False):
	episode_rewards = []
	for i in tqdm(range(num_episodes)):
		if adapt:
			ep_agent = deepcopy(agent)
			ep_agent.init_pad_optimizer()
		else:
			ep_agent = agent
		obs = env.reset()
		video.init(enabled=True)
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(ep_agent):
				action = ep_agent.select_action(obs)
			next_obs, reward, done, _ = env.step(action)
			video.record(env, eval_mode)
			episode_reward += reward
			if adapt:
				ep_agent.update_inverse_dynamics(*augmentations.prepare_pad_batch(obs, next_obs, action))
			obs = next_obs

		video.save(f'eval_{eval_mode}_{i}.mp4')
		episode_rewards.append(episode_reward)
		
		if L is not None:
			log_key = f'eval/episode_reward'
			if adapt:
				log_key += '_adapt'
			L.log(log_key, episode_reward, step)
			L.log('eval/episode', i, step)
			L.dump(step)

	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode,
		intensity=args.distracting_cs_intensity
	)

	# Set working directory
	work_dir = os.path.join(args.log_dir, args.algorithm, args.domain_name+'_'+args.task_name)
	print('Working directory:', work_dir)
	utils.make_dir(work_dir)

	# Create a dedicated directory for this eval mode's outputs
	eval_dir = os.path.join(work_dir, args.eval_mode)
	utils.make_dir(eval_dir)
	L = Logger(eval_dir)
	
	# 如果指定了模型路径则使用指定的模型
	if args.model_path and os.path.exists(args.model_path):
		model_path = args.model_path
		print(f"Using specified model: {model_path}")
		
		if not os.path.exists(work_dir):
			os.makedirs(work_dir)
			print(f"Created working directory: {work_dir}")
	else:
		# 原本的方式查找模型
		assert os.path.exists(work_dir), 'specified working directory does not exist'
		model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
		model_path = os.path.join(model_dir, str(args.train_steps)+'.pt')
	
	video_dir = utils.make_dir(os.path.join(eval_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

	# Check if evaluation has already been run
	if args.eval_mode == 'distracting_cs':
		results_fp = os.path.join(eval_dir, str(args.distracting_cs_intensity).replace('.', '_')+'.pt')
	else:
		results_fp = os.path.join(eval_dir, 'results.pt')
	
	if os.path.exists(results_fp):
		print(f'Results already exist for {args.eval_mode} in {eval_dir}, overwriting.')

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)
	
	# 加载模型
	print(f"Loading model from: {model_path}")
	agent = torch.load(model_path)
	agent.train(False)

	step = args.train_steps
	try:
		step = int(Path(model_path).stem)
	except ValueError:
		print(f'Could not parse step from {model_path}. Using `train_steps`: {step}.')

	print(f'\nEvaluating {eval_dir} for {args.eval_episodes} episodes (mode: {args.eval_mode})')
	reward = evaluate(env, agent, video, args.eval_episodes, args.eval_mode, L, step)
	print('Reward:', int(reward))

	adapt_reward = None
	if args.algorithm == 'pad':
		env = make_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed+42,
			episode_length=args.episode_length,
			action_repeat=args.action_repeat,
			mode=args.eval_mode
		)
		adapt_reward = evaluate(env, agent, video, args.eval_episodes, args.eval_mode, L, step, adapt=True)
		print('Adapt reward:', int(adapt_reward))

	# Save results
	torch.save({
		'args': args,
		'reward': reward,
		'adapt_reward': adapt_reward
	}, results_fp)
	print('Saved results to', results_fp)


if __name__ == '__main__':
	args = parse_args()
	main(args)


'''
python eval.py \
    --algorithm svea \
    --domain_name walker \
    --task_name walk \
    --seed 1 \
    --eval_mode color_hard \
    --train_steps 500000 \
    --log_dir /mnt/lustre/GPU4/home/wuhanpeng/dmcontrol-generalization-benchmark-main/logs/svea/svea_walker_walk_20241224_111447 \
    --save_video

'''