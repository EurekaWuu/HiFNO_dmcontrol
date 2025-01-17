from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.drqv2 import DrQV2Agent
from algorithms.svea import SVEA
from algorithms.hifno import HiFNOAgent
from algorithms.hifno_bisim import HiFNOBisimAgent
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
	'svea': SVEA,
	'hifno': HiFNOAgent,
	'hifno_bisim': HiFNOBisimAgent,
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
	elif args.algorithm == 'drqv2':
		return DrQV2Agent(
			obs_shape=obs_shape,
			action_shape=action_shape,
			args=args
		)
	return algorithm[args.algorithm](obs_shape, action_shape, args)