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
	'hifno_multigpu': HiFNOAgentMultiGPU,
    'svea_vis': SVEA_VIS
}


def make_agent(obs_shape, action_shape, args):
	if args.algorithm in ['hifno', 'hifno_multigpu']:
		# 根据算法名称选择对应的实现
		agent_class = algorithm[args.algorithm]
		agent = agent_class(
			obs_shape=obs_shape,
			action_shape=action_shape,
			args=args
		)
		agent = agent.to(torch.device('cuda:0'))
		return agent
	# 对于其他算法
	return algorithm[args.algorithm](obs_shape, action_shape, args)
