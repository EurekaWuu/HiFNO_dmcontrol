from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.drqv2 import DrQV2Agent
from algorithms.drqv2_notem import DrQV2NoTemAgent
from algorithms.drqv2_official import DrQV2OffAgent
from algorithms.svea import SVEA
from algorithms.hifno import HiFNOAgent
from algorithms.hifno_bisim import HiFNOBisimAgent
from algorithms.hifno_bisim_1 import HiFNOBisimAgent as HiFNOBisimAgent1
from algorithms.svea_vis import SVEA_VIS
from algorithms.pieg import PIEG

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'drqv2': DrQV2Agent,
	'drqv2_notem': DrQV2NoTemAgent,
	'drqv2_official': DrQV2OffAgent,
	'svea': SVEA,
	'hifno': HiFNOAgent,
	'hifno_bisim': HiFNOBisimAgent,
	'hifno_bisim_1': HiFNOBisimAgent1,
	'svea_vis': SVEA_VIS,
	'pieg': PIEG
}


def make_agent(obs_shape, action_shape, args):
	if args.algorithm == 'hifno':
		return HiFNOAgent(
			obs_shape=obs_shape,
			action_shape=action_shape,
			args=args
		)
	elif args.algorithm == 'hifno_bisim':
		return HiFNOBisimAgent(
			obs_shape=obs_shape,
			action_shape=action_shape,
			args=args
		)
	elif args.algorithm == 'hifno_bisim_1':
		return HiFNOBisimAgent1(
			obs_shape=obs_shape,
			action_shape=action_shape,
			args=args
		)
	elif args.algorithm == 'drqv2':
		return DrQV2Agent(
			obs_shape=obs_shape,
			action_shape=action_shape,
			args=args
		)
	elif args.algorithm == 'drqv2_notem':
		return DrQV2NoTemAgent(
			obs_shape=obs_shape,
			action_shape=action_shape,
			args=args
		)
	elif args.algorithm == 'drqv2_official':
		return DrQV2OffAgent(
			obs_shape=obs_shape,
			action_shape=action_shape,
			device=args.device if hasattr(args, 'device') else 'cuda',
			lr=args.lr,
			feature_dim=args.feature_dim,
			hidden_dim=args.hidden_dim,
			critic_target_tau=args.critic_target_tau,
			num_expl_steps=args.num_expl_steps,
			update_every_steps=args.update_every_steps,
			stddev_schedule=args.stddev_schedule,
			stddev_clip=args.stddev_clip,
			use_tb=args.use_tb,
			batch_size=args.batch_size,
			num_seed_frames=args.num_seed_frames,
			action_repeat=args.action_repeat
		)
	return algorithm[args.algorithm](obs_shape, action_shape, args)