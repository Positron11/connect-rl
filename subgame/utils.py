import pickle
import numpy as np
from collections import defaultdict


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


def load_table(filename:str) -> dict:
	"""Load Q-table from file."""
	
	with open(filename, "rb") as f:
		Q_table = defaultdict(lambda: [0.0] * 4, pickle.load(f))
	
	return Q_table