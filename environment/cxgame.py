from environment.cxenv import CXEnvironment


class CXGameAgent:
	"""Opponent agent base class for Connect-X game."""

	def __init__(self):
		self.env = None
	

	def play(self) -> tuple[int, tuple[bool, float]]:
		"""Play agent's turn."""

		pass


class CXGame:
	"""Play Connect-X game with user and agent."""

	def __init__(self, 
			  env:CXEnvironment, 
			  agent:CXGameAgent,
			  agent_player:int):
		
		self.env = env
		self.agent = agent
		self.agent_player = agent_player

		# set agent's environment
		self.agent.env = self.env


	def user_play(self) -> tuple[bool, float]:
		"""Get and play user's action."""

		while True:
			try:
				col = int(input(f"[turn] Your move (1-{self.env.cols}): ")) - 1
				return self.env.play(col)
			
			except Exception as e: print(f"[err.] Invalid move: {e}")


	def play(self) -> None:
		"""Play one game."""
		
		self.env.reset()

		while not self.env.game_over:
			player = self.env.current_player

			if player == self.agent_player:
				action, win, block_weight = self.agent.play() 
				print(f"[turn] Agent plays column {action + 1}")
				
			else: win, block_weight = self.user_play()

			# print board after agent moves or if game over
			if player == self.agent_player or self.env.game_over: print(f"\n{self.env}\n")

			# print block reward
			if block_weight: print(f"[info] {"Agent" if player == self.agent_player else "You"} blocked (weight = {block_weight})!")
		
		# print game result
		if win: print("[res.] Agent wins." if player == self.agent_player else "> You win.")
		else: print("[res.] It's a draw.")