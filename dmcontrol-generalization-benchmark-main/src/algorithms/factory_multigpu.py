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
from algorithms.hifno_bisim_1_multigpu import HiFNOBisimAgent as HiFNOBisimAgentMultiGPU
from algorithms.svea_vis import SVEA_VIS
import torch.distributed as dist

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
	'hifno': HiFNOAgent,
	'hifno_multigpu': HiFNOAgentMultiGPU,
	'hifno_bisim_1_multigpu': HiFNOBisimAgentMultiGPU,
	'svea_vis': SVEA_VIS
}


def make_agent(obs_shape, action_shape, args):
	if dist.is_initialized():
		world_size = dist.get_world_size()
		args.batch_size = args.batch_size * world_size

	if args.algorithm in ['hifno', 'hifno_multigpu', 'hifno_bisim_1_multigpu']:
		agent_class = algorithm[args.algorithm]
		
		agent = agent_class(
			obs_shape=obs_shape,
			action_shape=action_shape,
			args=args
		)
		return agent
	# 对于其他算法
	return algorithm[args.algorithm](obs_shape, action_shape, args)
