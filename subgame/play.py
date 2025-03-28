import sys
from collections import defaultdict

from environment.cxenv import CXEnvironment
from environment.cxgame import CXGame, CXGameAgent
from subgame.utils import select_action, load_table


# =============================================================== OPPONENT AGENT

class QTableGameAgent(CXGameAgent):
	"""Q-table opponent agent for Connect-4 game."""

	def __init__(self, Q_table:defaultdict):
		super().__init__()

		self.Q_table = Q_table
	

	def play(self) -> tuple[bool, float]:
		"""Play agent's turn."""

		state = self.env.state(self.env.current_player)
	
		valid_actions = self.env.valid_actions()
		action = select_action(valid_actions, self.Q_table[state])
				
		return action, *self.env.play(action)


# ==================================================================== PLAY GAME

if __name__ == "__main__":
	Q_table = load_table(sys.argv[1])

	# initialize 4x4 environment
	env = CXEnvironment(4, 4, 4)

	# initialize game with Q-table agent
	game = CXGame(env, QTableGameAgent(Q_table), 3 - int(sys.argv[2]))

	game.play()