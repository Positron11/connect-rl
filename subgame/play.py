import sys
import pickle
from collections import defaultdict

from environment.cxenv import CXEnvironment
from subgame.utils import select_action


# =============================================================== HELPER METHODS

def user_turn(env:CXEnvironment):
	"""Get and play user's action."""

	while True:
		try:
			col = int(input("Your move (1-4): ")) - 1
			return env.play(col)
		
		except Exception as e: print(f"Invalid move: {e}")


def ai_turn(env:CXEnvironment, player:int): # ----------------------------------
	"""Play action according to loaded policy."""

	state = env.state(player)
	
	valid_actions = env.valid_actions()
	action = select_action(valid_actions, Q[state])
	
	print(f"AI plays column {action + 1}")
	
	return env.play(action)


# ==================================================================== PLAY GAME

if __name__ == "__main__":
	# load Q-table
	with open(sys.argv[1], "rb") as f:
		Q = defaultdict(lambda: [0.0] * 4, pickle.load(f))

	# Initialize environment
	env = CXEnvironment(4, 4, 4)

	user = int(sys.argv[2])
	opponent = 3 - user

	# play one game
	while not env.game_over:
		player = env.current_player

		win, block_weight = user_turn(env) if player == user else ai_turn(env, player)

		# print board
		print(env)

		# print block reward
		if block_weight: print(f"{"You" if player == user else "AI"} blocked (weight = {block_weight})!")
	
	if win: print(f"{"You" if player == user else "AI"} win!")
	else: print("It's a draw.")