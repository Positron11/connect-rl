import numpy as np

class CXEnvironment:
	"""
	Game environment for a (non-trivial) Connect-X style game using a NumPy board.
	
	Parameters:
		rows (int): No. of rows in the board. Must be at least 3.
		cols (int): No. of columns in the board. Must be at least 3.
		x    (int): No. of contiguous pieces required to win. Must be at least 3 and cannot exceed both no. of rows and columns.
	"""

	def __init__(self, rows:int, cols:int, x:int):
		if rows < 3 or cols < 3: 
			raise ValueError("Invalid number of rows/columns.")
		
		# board dimensions
		self.rows = rows
		self.cols = cols
		
		if x < 3 or (x > rows and x > cols): 
			raise ValueError("Invalid number of rows/columns.")

		# no. of contiguous pieces to win
		self.x = x

		# start in initial state 
		self.reset()


	def __str__(self) -> str:
		"""Print the board state."""

		# visual state representation
		visual = ["○", "\033[33m●\033[0m", "\033[31m●\033[0m"]

		return "\n".join((" ".join(visual[i] for i in row)) for row in self.board)


	def reset(self) -> None:
		"""Reset board to initial state."""
		
		# set all holes to empty 
		self.board = np.zeros((self.rows, self.cols), dtype=int)

		# game status
		self.game_over = False
		
		# start with player 1
		self.current_player = 1

	
	def is_valid(self, row:int, col:int) -> bool:
		"""Check if (row, col) is a valid coordinate in the board."""
		
		return  0 <= row < self.rows and 0 <= col < self.cols


	def get_free(self) -> list:
		"""Get the next free position in each column."""
		
		indices = [np.where(c == 0)[0] for c in self.board.T]
		return [int(i[-1]) if len(i) else -1 for i in indices]
	
	
	def valid_actions(self) -> list:
		"""Get list of valid column indices."""

		return [i for i, r in enumerate(self.get_free()) if r != -1]
	

	def board_full(self) -> bool:
		"""Check if any valid moves left."""

		return self.board[0].all()


	def play(self, col) -> tuple[int, bool]:
		"""Place the piece in the selected column."""

		if self.game_over: raise ValueError("Invalid move - game is over.")

		# try to play valid move for selected column
		if (row := self.get_free()[col]) == -1: raise ValueError("Invalid move - column is full.")
		else: self.board[row][col] = self.current_player

		win, block = self.check_win(row, col)
		self.game_over = win or self.board_full()

		# switch player
		self.current_player = 3 - self.current_player

		return win, block


	def check_win(self, row, col) -> tuple[bool, float]:
		"""
		Check if placing a piece at (row, col) caused a win.
		
		Returns:
			bool: The last move won the game
			float: Block reward weight based on aggregate blocks 
		"""
		
		# define directions: horizontal, vertical, diagonal
		directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
		
		player = self.board[row, col]
		opp = 3 - player
		
		# check for win
		for dr, dc in directions:
			player_count = 1 
			
			# positive direction.
			r, c = row + dr, col + dc
			while self.is_valid(r, c) and self.board[r, c] == player:
				player_count += 1
				r += dr
				c += dc

			# negative direction.
			r, c = row - dr, col - dc
			while self.is_valid(r, c) and self.board[r, c] == player:
				player_count += 1
				r -= dr
				c -= dc
			
			# return won, but not blocked (not significant)
			if player_count >= self.x: return True, 0

		# check for block
		block_reward_weight = 0
		single_partial_blocks = 0

		for dr, dc in directions:
			opp_count = 0
			
			# positive direction.
			r, c = row + dr, col + dc
			while self.is_valid(r, c) and self.board[r, c] == opp:
				opp_count += 1
				r += dr
				c += dc

			# check if sequence ends on open, playable hole
			open_pos = self.is_valid(r, c) and self.get_free()[c] == r

			# negative direction.
			r, c = row - dr, col - dc
			while self.is_valid(r, c) and self.board[r, c] == opp:
				opp_count += 1
				r -= dr
				c -= dc

			# check if sequence ends on open, playable hole
			open_neg = self.is_valid(r, c) and self.get_free()[c] == r
			
			# immediate complete block
			if opp_count == self.x - 1: block_reward_weight += 1
			
			# partial blocks
			elif opp_count == self.x - 2:
				if open_neg and open_pos: block_reward_weight += 1
				if open_neg != open_pos: single_partial_blocks += 1

		# consider only multiple single open-ended partial blocks
		if single_partial_blocks > 1: block_reward_weight += single_partial_blocks * 0.5
		return False, block_reward_weight
	

	def state(self, player:int) -> tuple[int, tuple]:
		"""Return the Q-table index corresponding to the state."""

		return player, tuple(self.board.flatten())
