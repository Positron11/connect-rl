from collections import defaultdict

from environment.cxenv import CXEnvironment
from subgame.utils import select_action

class WindowAgent:
	"""Sliding-window connext-X player."""

	def __init__(self, Q_table:defaultdict):
		self.Q = Q_table


	def extract_windows(self, env:CXEnvironment):
		"""Extract 4x4 windows from larger board."""

		# flip environment board indexing
		board = env.board

		windows = []

		# columns 0 to 3...
		for c in range(board.shape[1] - 3):
			# rows 5 to 3...
			for r in range(board.shape[0] - 1, 2, -1):
				# extract window 
				window = board[r-3:r+1, c:c+4]
				
				# check for (partially) vacant or last available
				if (not window.all()) or (r == 3): 
					# flip indexing back
					windows.append(window)
					break

		return windows

		
	def get_action(self, env:CXEnvironment):
		"""Get agent's optimal action in current game state."""

		# extract windows
		windows = self.extract_windows(env)

		valid_actions = env.valid_actions()

		window_actions = []

		for i, window in enumerate(windows):
			q_values = self.Q[(env.current_player, tuple(window.flatten()))]
			
			# select best action
			action = select_action(valid_actions, q_values)

			window_actions.append((action, q_values[action]))

		return max(window_actions, key=lambda t: t[1])[0]
			




