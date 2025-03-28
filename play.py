import sys

from environment.cxenv import CXEnvironment
from environment.cxgame import CXGame, CXGameAgent
from subgame.utils import load_table
from winagent.agent import WindowAgent


# =============================================================== OPPONENT AGENT

class WindowGameAgent(CXGameAgent):
	"""Q-table opponent agent for Connect-4 game."""

	def __init__(self, agent:WindowAgent):
		super().__init__()

		self.agent = agent
	

	def play(self) -> tuple[bool, float]:
		"""Play agent's turn."""

		action = agent.get_action(self.env)
				
		return action, *self.env.play(action)

	
# ==================================================================== PLAY GAME

if __name__ == "__main__":
	Q_table = load_table(sys.argv[1])

	# create window agent
	agent = WindowAgent(Q_table)

	# initialize 6x7 environment
	env = CXEnvironment(6, 7, 4)

	# initialize game with window agent
	game = CXGame(env, WindowGameAgent(agent), 3 - int(sys.argv[2]))

	game.play()