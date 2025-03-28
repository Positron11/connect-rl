import sys
import pickle
import random
from collections import defaultdict

from environment.cxenv import CXEnvironment
from subgame.utils import select_action


# =============================================================== HELPER METHODS

def test_qtable(Q:dict, games:int = 1000):
	"""Test Q-table against random opponent for a batch of games."""

	wins = 0
	losses = 0
	draws = 0
	
	for i_game in range(games):
		env = CXEnvironment(4, 4, 4)
		
		# alternate opponent (starting with player 1)
		opponent = (i_game % 2) + 1
		agent = 3 - opponent
				
		# play game
		while not env.game_over:
			player = env.current_player
			
			valid_actions = env.valid_actions()
			
			# agent"s turn
			if player == agent:	
				state = env.state(agent)
				action = select_action(valid_actions, Q[state])
			
			# opponent"s turn
			else: action = random.choice(valid_actions)
			
			win, _ = env.play(action)
		
		# game result
		if win:
			if player == agent: wins += 1
			elif player == opponent: losses += 1
		
		else: draws += 1

	return wins, losses, draws


# ===================================================================== RUN TEST

if __name__ == "__main__":
	games = int(sys.argv[2])

	# load Q-table
	with open(sys.argv[1], "rb") as f:
		q_table = defaultdict(lambda: [0.0] * 4, pickle.load(f))

	wins, losses, draws = test_qtable(q_table, games)
	
	print(f"Results over {games} games:")
	print(f"Wins: {wins} ({wins/games:.3f})")
	print(f"Losses: {losses} ({losses/games:.3f})")
	print(f"Draws: {draws} ({draws/games:.3f})")