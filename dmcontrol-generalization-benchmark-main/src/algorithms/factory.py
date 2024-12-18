import torch
from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.hifno import HiFNOAgent
from algorithms.hifno_multigpu import HiFNOAgent as HiFNOAgentMultiGPU
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
    'svea_vis': SVEA_VIS
}


def make_agent(obs_shape, action_shape, args):
	if args.algorithm == 'hifno':
		if torch.cuda.device_count() > 1:
			return HiFNOAgentMultiGPU(obs_shape, action_shape, args)
		else:
			return HiFNOAgent(obs_shape, action_shape, args)
	else:
		raise NotImplementedError
