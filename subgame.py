import os
import pickle
import random
import numpy as np
from collections import defaultdict
from cxenv import CXEnvironment


SAVE_PATH = "qtables"
os.makedirs(SAVE_PATH, exist_ok=True)


# Generational hyperparameters
GENERATIONS 		= 5
GAMES 				= 1000
PROMOTE_THRESHOLD 	= 0.15
SHARPNESS_TOLERANCE	= 0.01

# Q-learning hyperparameters
ALPHA		= 0.2   # learning rate
GAMMA 		= 0.95	# discount factor
PLAY_SPLIT	= 0.5	# frozen vs. self-play
EPISODES	= 100000

# rewards
BLOCK_REWARD	= 0.3
WIN_REWARD		= 10
LOSE_REWARD		= -3


def choose_action(valid_actions:list, q_vals:list) -> int:
	"""Choose action with slight stochasticity among top Q-values."""
	
	# mask Q-values corr. to invalid actions
	masked_q = [(q if i in valid_actions else -np.inf) for i, q in enumerate(q_vals)]

	# choose best actions (within tolerance of max)
	max_q = max(masked_q)
	candidate_actions = [i for i, q in enumerate(masked_q) if np.isclose(q, max_q, atol=1e-5)]

	return int(np.random.choice(candidate_actions))


def qlearn(frozen_Q:defaultdict=None, opp:int=1) -> defaultdict:
	"""
	Trains a new Q-table via tabular Q-learning.

	Parameters:
		frozen_Q (defaultdict): Q-table of the frozen opponent. If None, self-play is used.
		opp (int): Player index (1 or 2) the frozen opponent is playing as.

	Returns:
		defaultdict: Q-table of the newly trained agent
	"""

	# init. Q-table
	Q = defaultdict(lambda: [0.0] * 4) 

	# create win counter
	wins = [0] * 3
		
	# init. environment
	env = CXEnvironment(4, 4, 4)

	for e in range(EPISODES):
		# print episodic log
		if (e + 1) % (EPISODES // 5) == 0: 
			win_rate = wins[3 - opp] / max(e, 1)
			draw_rate = wins[0] / max(e, 1)
			print(f"> Episode {e}: Win rate = {win_rate:.3f}, Draw rate = {draw_rate:.3f}")
			
		# epsilon decay
		epsilon = max(0.1, 1.0 - e / (EPISODES * 0.75))

		# split of self-play or vs. opponent
		opp_Q = frozen_Q if random.random() < PLAY_SPLIT else None
		
		done = False
		state = env.canonicalized()

		while not done:
			player = env.current_player

			valid_actions = env.valid_actions()

			# opponent's turn
			if opp_Q and player == opp: action = choose_action(valid_actions, opp_Q[state])

			else:
				# epsilon-greedy action selection
				if random.random() < epsilon: action = random.choice(valid_actions)
				else: action = choose_action(valid_actions, Q[state])

			done, win, block_weight = env.play(action)
			
			next_state = env.canonicalized()
			next_player = env.current_player

			# determine reward (win: 1, lose: -1, draw: 0)
			if done and win: reward = WIN_REWARD if player != opp else LOSE_REWARD
			elif player != opp: reward = BLOCK_REWARD * block_weight
			else: reward = 0

			# Minimax
			if done: best_next = 0
			elif next_player == opp: best_next = min(Q[next_state])
			else: best_next = max(Q[next_state])

			# Q-learning update
			Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])

			state = next_state

		wins[player if win else 0] += 1

		env.reset()

	return Q


def evaluate(new_Q:defaultdict, frozen_Q:defaultdict, frozen_player:int=1) -> tuple[float, float]:
	"""
	Evaluates the performance of a newly trained agent against a frozen opponent.

	Parameters:
		new_Q (defaultdict): Q-table of the newly trained agent.
		frozen_Q (defaultdict): Q-table of the frozen opponent.
		frozen_player (int): The player index (1 or 2) the frozen agent is playing as.

	Returns:
		tuple:
			win_rate (float): Proportion of games won by the learner.
			sharpness (float): Average sharpness (confidence) of learner's Q-values.
	"""

	wins = [0] * 3
	sharpness_vals = []

	for g in range(GAMES):
		env = CXEnvironment(4, 4, 4)
		done = False

		first_player = 2 if g % 2 == 1 else 1 
		env.current_player = first_player

		while not done:
			valid_actions = env.valid_actions()

			# extract valid Q-values
			Q = frozen_Q if env.current_player == frozen_player else new_Q
			q_values = [(q if i in valid_actions else -np.inf) for i, q in enumerate(Q[env.canonicalized()])]

			# compute sharpness
			pos_q_values = [v for v in q_values if v != -np.inf]
			sharpness = max(pos_q_values) - np.mean(pos_q_values) if q_values else 0
			sharpness_vals.append(sharpness)

			# play action
			candidate_actions = [i for i, q in enumerate(q_values) if np.isclose(q, max(q_values), atol=1e-5)]
			action = int(np.random.choice(candidate_actions))

			done, win, _ = env.play(action)

		if win and env.current_player == frozen_player:
			wins[1 if first_player != env.current_player else 2] += 1
		
		else: wins[0] += 1

	# get learner's win and draw rate
	win_rate = [wins[1] / GAMES, wins[2] / GAMES]
	draw_rate = wins[0] / GAMES

	sharpness_avg = np.mean(sharpness_vals)

	return win_rate, draw_rate, sharpness_avg


frozen_Q = None
frozen_player = 1


# main loop
for g in range(GENERATIONS):
	print(f"\n[Train] Gen. {g} - Learner as Player {3 - frozen_player}")
	
	new_Q = qlearn(frozen_Q, frozen_player)

	if frozen_Q:
		win_rate, draw_rate, sharp_new = evaluate(new_Q, frozen_Q, frozen_player)

		print(f"[Eval] Gen. {g-1} vs. {g}: Win rate = {win_rate[0]:.3f}/{win_rate[1]:.3f}, Draw rate = {draw_rate:.3f}, Sharpness = {sharp_new:.4f} (Prev. = {sharp_old:.4f})")

		# promote generation if passes conditions
		if sum(win_rate) >= PROMOTE_THRESHOLD:
			print("> Promoted to new baseline")

			frozen_Q = new_Q
			frozen_player = 3 - frozen_player
			
			sharp_old = sharp_new

		else: print("> Rejected - keeping previous gen.")

	# auto-promote first generation
	else: 
		frozen_Q = new_Q
		frozen_player = 3 - frozen_player
		
		sharp_vals = [max(v) - np.mean(v) for v in frozen_Q.values() if any(v)]
		sharp_old = np.mean(sharp_vals) if sharp_vals else 0.0

	# Save the current generation
	with open(f"{SAVE_PATH}/g{g}-qtable_4x4.pkl", "wb") as f:
		pickle.dump(dict(new_Q), f)
