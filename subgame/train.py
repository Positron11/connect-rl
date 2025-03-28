import os
import pickle
import random
import datetime
import numpy as np
from copy import deepcopy
from collections import defaultdict

from environment.cxenv import CXEnvironment
from subgame.utils import select_action
from subgame.test import test_qtable


# ============================================================== HYPERPARAMETERS

SAVE_PATH = "generated/tables"

EPSILON_START = 0.8
EPSILON_FLOOR = 0.05
EPSILON_DECAY = 0.9
EPSILON_BOOST = 1.01

EPISODES	= 200000
GAMMA 		= 0.9
PLAY_SPLIT	= 0.7
ALPHA		= 0.05

WIN_REWARD	 = 1.0
LOSE_REWARD	 = -1.0
DRAW_REWARD  = 0.0
BLOCK_REWARD = 0.1

LOGGING_INTERVAL = 5000
EPSILON_UPDATE_INTERVAL = 5000


# ======================================================= MODEL TRAINING METHODS

def compute_reward(env:CXEnvironment,
				   player:int, 
				   opponent:int, 
				   win:bool, 
				   block_weight:float) -> float:
	
	"""Compute reward based on game outcome."""

	if env.game_over:
		# we have a winner (from learner's perespective)
		if win: reward = LOSE_REWARD if player == opponent else WIN_REWARD
		
		# game ends in a draw
		else: reward = DRAW_REWARD
		
	# game not over, and learner has blocked
	elif player != opponent and block_weight > 0: reward = BLOCK_REWARD * block_weight

	else: reward = 0

	return reward


def q_update(Q:defaultdict, # --------------------------------------------------
			 state:tuple, 
			 action:int, 
			 reward:float, 
			 next_state:tuple, 
			 game_over: bool,
			 alpha:float = ALPHA, 
			 gamma:float = GAMMA) -> None:
	
	"""Q-learning update step."""

	best_next = max(Q[next_state]) if not game_over else 0

	# Q-learning update
	Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])


def play_game(env:CXEnvironment, # ---------------------------------------------
			  Q:defaultdict,
			  opponent:int, 
			  adversarial:bool, 
			  epsilon:float,
			  frozen_Q:defaultdict = None) -> int:
	
	"""Play a game of Connect-X with Q-learning."""

	# explicitly compute learner
	learner = 3 - opponent
	
	# cumulative reward for current episode
	total_reward = 0

	state = env.state(learner)

	while not env.game_over:
		player = env.current_player

		valid_actions = env.valid_actions()

		# opponent's turn - random if no frozen Q-table
		if adversarial and player == opponent: 
			if frozen_Q: action = select_action(valid_actions, frozen_Q[state], True)
			else: action = random.choice(valid_actions)

		# learner's turn - epsilon-greedy action selection
		else:
			if random.random() < epsilon: action = random.choice(valid_actions)
			else: action = select_action(valid_actions, Q[state], True)

		# play move and get result
		game_result = env.play(action)

		reward = compute_reward(env, player, opponent, *game_result)
		total_reward += reward

		next_state = env.state(learner)

		# perform Q-learning update
		q_update(Q, state, action, reward, next_state, env.game_over)

		state = next_state

	return total_reward


def epsilon_update(epsilon:float, # --------------------------------------------
				   current_avg_reward:float,
				   best_avg_reward:float,
				   epsilon_start:float = EPSILON_START,
				   epsilon_floor:float = EPSILON_FLOOR,
				   epsilon_decay:float = EPSILON_DECAY,
				   epsilon_boost:float = EPSILON_BOOST) -> float:
	
	"""Update epsilon value based on performance."""

	# conditional decay
	epsilon *= epsilon_decay if current_avg_reward - best_avg_reward > -5E-3 else epsilon_boost
	
	# return bounded epsilon
	return max(epsilon_floor, min(epsilon, epsilon_start))


def qlearn(base_Q:defaultdict	= None, # --------------------------------------
		   frozen_Q:defaultdict	= None, 
		   epsilon_start:float	= EPSILON_START,
		   play_split:float		= PLAY_SPLIT) -> defaultdict:
	
	"""Train a new Q-table via tabular Q-learning."""

	mode_info = ("Adversarial" if frozen_Q else "Random") if play_split else "Self-Play"
	split_info = f"(Split = {play_split:.2f})" if play_split else ""

	# print training header
	print(f"\n[train] Tabular Q-Learning - {mode_info} {split_info}")
		
	# init. 4x4 environment
	env = CXEnvironment(4, 4, 4)

	# load/init. learner Q-table
	Q = base_Q or defaultdict(lambda: [0.0] * env.cols)

	epsilon = epsilon_start

	# reward trackers
	rewards = []
	best_avg_reward = -np.inf

	for i_episode in range(EPISODES):	
		env.reset()

		# self-play or vs. opponent
		adversarial = random.random() < play_split

		# alternate opponent (starting with player 1)
		opponent = (i_episode % 2) + 1
		
		# play game and get learner's total reward
		episodic_reward = play_game(env, Q, opponent, adversarial, epsilon)

		rewards.append(episodic_reward)

		# periodic actions
		if (i_episode > 0):
			if (i_episode % EPSILON_UPDATE_INTERVAL == 0):
				avg_reward = np.mean(rewards[-EPSILON_UPDATE_INTERVAL:])
				
				# conditionally update epsilon
				epsilon = epsilon_update(epsilon, avg_reward, best_avg_reward)
				
				# update best avg. periodic reward
				best_avg_reward = max(avg_reward, best_avg_reward)
			
			# log progress
			if (i_episode % LOGGING_INTERVAL == 0):
				avg_reward = np.mean(rewards[-LOGGING_INTERVAL:])
				print(f"- Ep. {i_episode} - Avg. Reward = {np.mean(rewards[-LOGGING_INTERVAL:]):.3f}, Epsilon = {epsilon:.4f}")
		
	return Q


def evaluate(Q:defaultdict, games:int=1000): # ---------------------------------
	"""Test a Q-table against random opponent."""

	print("\n[test] Testing Q-table...")

	wins, losses, draws = test_qtable(Q, games)

	print(f" - Win rate: {wins/games:.3f}")
	print(f" - Loss rate: {losses/games:.3f}")
	print(f" - Draw rate: {draws/games:.3f}")


# ================================================================ TRAINING LOOP

if __name__ == "__main__":
	# welcome statement
	print(f"""\n4x4 Subgame - TQL {datetime.datetime.now()}

[info] Static hyperparameters:
 - Episodes = {EPISODES}
 - Gamma    = {GAMMA}
 - Alpha    = {ALPHA}

[info] Rewards:
- Win   = {WIN_REWARD}
- Lose  = {LOSE_REWARD}
- Draw  = {DRAW_REWARD}
- Block = {BLOCK_REWARD}""")

	# file environment
	os.makedirs(SAVE_PATH, exist_ok=True)

	# phase 1 - random opponent
	trained_Q_random = qlearn(play_split=1.0)
	evaluate(trained_Q_random)

	# phase 2 - pure self-play
	trained_Q_selfplay = qlearn(base_Q=deepcopy(trained_Q_random), play_split=0.0)
	evaluate(trained_Q_selfplay)

	# phase 3 - adversarial
	Q = deepcopy(trained_Q_selfplay)
	
	for i_generation in range(5):
		Q = qlearn(base_Q=deepcopy(Q), frozen_Q=Q)
		evaluate(Q)

	# save the final generation
	print("\n[info] Saved table to:", SAVE_PATH)

	with open(f"{SAVE_PATH}/q_table_4x4.pkl", "wb") as f:
		pickle.dump(dict(Q), f)
