import argparse
import numpy as np
import os
import utils

def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='walker')
	parser.add_argument('--task_name', default='walk')
	parser.add_argument('--frame_stack', default=3, type=int)
	parser.add_argument('--action_repeat', default=4, type=int)
	parser.add_argument('--episode_length', default=1000, type=int)
	parser.add_argument('--eval_mode', default='color_hard', type=str)
	
	# agent
	parser.add_argument('--algorithm', default='sac', type=str)
	parser.add_argument('--train_steps', default='500k', type=str)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=0, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# auxiliary tasks
	parser.add_argument('--aux_lr', default=1e-3, type=float)
	parser.add_argument('--aux_beta', default=0.9, type=float)
	parser.add_argument('--aux_update_freq', default=2, type=int)

	# soda
	parser.add_argument('--soda_batch_size', default=256, type=int)
	parser.add_argument('--soda_tau', default=0.005, type=float)

	# svea
	parser.add_argument('--svea_alpha', default=0.5, type=float)
	parser.add_argument('--svea_beta', default=0.5, type=float)

	# eval
	parser.add_argument('--save_freq', default='100k', type=str)
	parser.add_argument('--eval_freq', default='10k', type=str)
	parser.add_argument('--eval_episodes', default=30, type=int)
	parser.add_argument('--distracting_cs_intensity', default=0., type=float)

	# misc
	parser.add_argument('--seed', default=None, type=int)
	parser.add_argument('--log_dir', default='logs', type=str)
	parser.add_argument('--save_video', default=False, action='store_true')

	# HiFNO specific parameters
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--critic_target_tau', default=0.01, type=float)
	parser.add_argument('--num_scales', default=3, type=int)
	parser.add_argument('--embed_dim', default=64, type=int)
	parser.add_argument('--depth', default=2, type=int)
	parser.add_argument('--patch_size', default=4, type=int)

	# 双模拟相关参数
	parser.add_argument('--lambda_BB', type=float, default=0.8,
						help='Base Bisimulation loss weight')
	parser.add_argument('--lambda_ICC', type=float, default=0.4,
						help='Inter-context Consistency loss weight')
	parser.add_argument('--lambda_CC', type=float, default=0.4,
						help='Cross Consistency loss weight')
	parser.add_argument('--bisim_p', type=int, default=2,
						help='Norm used in bisimulation distance calculation')

	# drqv2 相关参数
	parser.add_argument('--feature_dim', default=50, type=int,
						help='Feature dimension used by encoders')
	parser.add_argument('--update_every_steps', default=2, type=int,
						help='Training frequency')
	parser.add_argument('--replay_dir', default='replay',
						help='Directory to store replay data')
	parser.add_argument('--num_workers', default=4, type=int,
						help='Number of workers for replay data loading')
	parser.add_argument('--nstep', default=3, type=int,
						help='Number of steps for n-step returns')
	parser.add_argument('--stddev_schedule', default='linear(1.0,0.1,500000)',
						type=str,
						help='Schedule for action std')
	parser.add_argument('--stddev_clip', default=0.3, type=float,
						help='Clip range for action std')
	parser.add_argument('--num_expl_steps', default=2000, type=int,
						help='Exploration steps')
	parser.add_argument('--use_tb', default=True, action='store_true',
						help='Whether to record training logs e.g. in TensorBoard')
	parser.add_argument('--num_seed_frames', default=4000, type=int,
						help='Number of frames to collect before training（用于drqv2_official）')

	# hifno_bisim_1
	parser.add_argument('--lambda_SC', type=float, default=0.5,
						help='Semantic Consistency loss weight')
	parser.add_argument('--lambda_clip', type=float, default=0.5,
						help='CLIP guided loss weight')
	parser.add_argument('--clip_loss_weight', type=float, default=0.5,
						help='Overall CLIP loss weight')
	parser.add_argument('--use_clip_guided_bisim', type=bool, default=True,
						help='Whether to use CLIP guided bisimulation loss')

	parser.add_argument('--use_sc_loss', type=utils.str2bool, default=True,
						help='是否使用语义类内一致性损失')
	parser.add_argument('--use_clip_bisim_loss', type=utils.str2bool, default=True,
						help='是否使用CLIP引导的双模拟损失')

	parser.add_argument('--model_path', default=None, type=str, help='直接指定模型文件路径，不通过目录结构查找')

	args = parser.parse_args()

	assert args.algorithm in {
		'sac', 'rad', 'curl', 'pad', 'soda', 'drq', 'drqv2', 'drqv2_notem', 'drqv2_official',
		'svea', 'hifno', 'hifno_multigpu', 'hifno_bisim', 'hifno_bisim_1', 'hifno_bisim_1_multigpu', 'pieg'
	}, f'specified algorithm "{args.algorithm}" is not supported'

	assert args.eval_mode in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode}" is not supported'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	intensities = {0., 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
	assert args.distracting_cs_intensity in intensities, f'distracting_cs has only been implemented for intensities: {intensities}'

	args.train_steps = int(args.train_steps.replace('k', '000'))
	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))

	if args.eval_mode == 'none':
		args.eval_mode = None

	if args.algorithm in {'rad', 'curl', 'pad', 'soda'}:
		args.image_size = 100
		args.image_crop_size = 84
	else:
		args.image_size = 84
		args.image_crop_size = 84
	
	
	if args.algorithm == 'drqv2_official':
		os.makedirs(args.replay_dir, exist_ok=True)

	return args
