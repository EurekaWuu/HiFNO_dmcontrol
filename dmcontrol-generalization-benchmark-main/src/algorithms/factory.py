from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.hifno import HiFNOAgent
from algorithms.hifno_bisim import HiFNOBisimAgent
from algorithms.svea_vis import SVEA_VIS

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
	'hifno': HiFNOAgent,
	'hifno_bisim': HiFNOBisimAgent,
	'svea_vis': SVEA_VIS
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
	return algorithm[args.algorithm](obs_shape, action_shape, args)