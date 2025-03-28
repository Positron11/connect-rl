import numpy as np


def select_action(valid_actions:list[int], q_vals:list[float], stochastic:bool=False) -> int:
	"""Choose action with optional stochasticity."""
	
	# mask Q-values corr. to invalid actions
	masked_q = [(q if i in valid_actions else -np.inf) for i, q in enumerate(q_vals)]

	if stochastic:
		max_q = max(masked_q)
		
		# choose best actions (within tolerance of max)
		candidate_actions = [i for i, q in enumerate(masked_q) if np.isclose(q, max_q, atol=1e-5)]

		return int(np.random.choice(candidate_actions))
	
	else: return int(np.argmax(masked_q))