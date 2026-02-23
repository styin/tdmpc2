import numpy as np
import gymnasium as gym
import torch
from torchvision.transforms import functional as F

from envs.wrappers.timeout import Timeout

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.env.camera_name = "corner2"
		self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
		self.env._freeze_rand_vec = False

		# Domain randomization setup
		self._domain_rand = cfg.get('domain_rand', False)
		if self._domain_rand:
			self._obj_body_id = self.env.model.body('obj').id
			self._obj_geom_id = self.env.model.geom('objGeom').id
			# Store original values for restoring if toggled off
			self._default_mass = self.env.model.body_mass[self._obj_body_id].copy()
			self._default_inertia = self.env.model.body_inertia[self._obj_body_id].copy()
			self._default_friction = self.env.model.geom_friction[self._obj_geom_id].copy()
			self._default_size = self.env.model.geom_size[self._obj_geom_id].copy()
			# Ranges from config
			self._mass_range = cfg.get('obj_mass_range', [0.5, 1.5])
			self._friction_range = cfg.get('obj_friction_range', [0.5, 1.5])
			self._size_range = cfg.get('obj_size_range', [0.015, 0.03])

	def _randomize_domain(self):
		"""Randomize object physics parameters at the start of each episode."""
		# Mass (and scale inertia proportionally)
		new_mass = np.random.uniform(*self._mass_range)
		mass_scale = new_mass / self._default_mass
		self.env.model.body_mass[self._obj_body_id] = new_mass
		self.env.model.body_inertia[self._obj_body_id] = self._default_inertia * mass_scale
		# Sliding friction (index 0 of geom_friction triplet)
		new_slide_friction = np.random.uniform(*self._friction_range)
		friction = self._default_friction.copy()
		friction[0] = new_slide_friction
		self.env.model.geom_friction[self._obj_geom_id] = friction
		# Cylinder radius (index 0 of geom_size); keep height unchanged
		new_radius = np.random.uniform(*self._size_range)
		size = self._default_size.copy()
		size[0] = new_radius
		self.env.model.geom_size[self._obj_geom_id] = size

	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(info.get('success', 0.0)),
		}
		info['score'] = info['success']
		return info

	def reset(self, **kwargs):
		if self._domain_rand:
			self._randomize_domain()
		super().reset(**kwargs)
		obs, _, _, _, info = self.env.step(
			np.zeros(self.env.action_space.shape, dtype=np.float32)
		)
		obs = obs.astype(np.float32)
		return obs, self._extract_info(info)

	def step(self, action):
		reward = 0
		terminated = False
		truncated = False
		info = {}
		for _ in range(2):
			obs, r, terminated, truncated, info = self.env.step(action.copy())
			reward += r
			if terminated or truncated:
				break
		obs = obs.astype(np.float32)
		info['terminated'] = terminated
		info['truncated'] = truncated
		return obs, reward, terminated, truncated, self._extract_info(info)

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		height = kwargs.get('height', 224)
		width = kwargs.get('width', 224)
		frame = torch.from_numpy(self.env.render().copy()).permute(2, 0, 1)
		frame = frame.flip(1)
		frame = F.resize(frame, (height, width)).permute(1, 2, 0).numpy()
		return frame

	def close(self):
		self.env.close()


def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-observable"
	if not cfg.task.startswith('mw-') or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
		raise ValueError('Unknown task:', cfg.task)
	env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](
		seed=cfg.seed,
		render_mode='rgb_array'
	)
	env = MetaWorldWrapper(env, cfg)
	if cfg.obs == 'rgb':
		from envs.wrappers.pixels import Pixels
		env = Pixels(env, cfg)
	env = Timeout(env, max_episode_steps=100)
	return env
