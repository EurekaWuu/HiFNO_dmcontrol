from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.hifno import HiFNOAgent

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA
}


def make_agent(obs_shape, action_shape, args):
	if args.algorithm == 'hifno':
		return HiFNOAgent(
			obs_shape=obs_shape,
			action_shape=action_shape,
			args=args
		)
	return algorithm[args.algorithm](obs_shape, action_shape, args)
