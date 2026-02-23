import gymnasium as gym


class Timeout(gym.Wrapper):
	"""
	Wrapper for enforcing a time limit on the environment.
	"""

	def __init__(self, env, max_episode_steps):
		super().__init__(env)
		self._max_episode_steps = max_episode_steps
	
	@property
	def max_episode_steps(self):
		return self._max_episode_steps

	def reset(self, **kwargs):
		self._t = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		result = self.env.step(action)
		if isinstance(result, tuple) and len(result) == 5:
			obs, reward, terminated, truncated, info = result
			done = terminated or truncated
		else:
			obs, reward, done, info = result
		self._t += 1
		timed_out = self._t >= self.max_episode_steps
		done = done or timed_out
		if isinstance(result, tuple) and len(result) == 5:
			truncated = truncated or timed_out
			return obs, reward, terminated, truncated, info
		return obs, reward, done, info
