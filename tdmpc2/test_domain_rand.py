"""
Dry-run test for domain randomization on mw-push.
Creates the MetaWorld environment with domain_rand=True and runs a few episodes
to verify that object mass, friction, and size are randomized each reset.
"""
import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')

import numpy as np
from termcolor import colored

from envs.metaworld import make_env, MetaWorldWrapper


class FakeConfig:
    """Minimal config object to drive make_env without Hydra."""
    def __init__(self, **kwargs):
        defaults = dict(
            task='mw-push',
            obs='state',
            seed=1,
            domain_rand=True,
            obj_mass_range=[0.5, 1.5],
            obj_friction_range=[0.5, 1.5],
            obj_size_range=[0.015, 0.03],
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def find_metaworld_wrapper(env):
    """Walk the wrapper chain to locate the MetaWorldWrapper."""
    while env is not None:
        if isinstance(env, MetaWorldWrapper):
            return env
        env = getattr(env, 'env', None)
    return None


def main():
    num_episodes = 5
    steps_per_episode = 10

    # ---- Build env with domain_rand=True ----
    cfg = FakeConfig(domain_rand=True)
    env = make_env(cfg)
    mw = find_metaworld_wrapper(env)
    assert mw is not None, "MetaWorldWrapper not found in wrapper chain"
    assert mw._domain_rand, "domain_rand flag not set"

    print(colored("\n===== Domain Randomization Dry-Run: mw-push =====\n", 'cyan', attrs=['bold']))
    print(f"Default mass    : {mw._default_mass}")
    print(f"Default friction: {mw._default_friction}")
    print(f"Default size    : {mw._default_size}")
    print(f"Mass range      : {mw._mass_range}")
    print(f"Friction range  : {mw._friction_range}")
    print(f"Size range      : {mw._size_range}")
    print()

    obj_body_id = mw._obj_body_id
    obj_geom_id = mw._obj_geom_id

    masses, frictions, sizes = [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        # Read the randomized physics params right after reset
        cur_mass = float(mw.env.model.body_mass[obj_body_id])
        cur_friction = float(mw.env.model.geom_friction[obj_geom_id][0])
        cur_size = float(mw.env.model.geom_size[obj_geom_id][0])
        masses.append(cur_mass)
        frictions.append(cur_friction)
        sizes.append(cur_size)

        print(colored(f"Episode {ep+1}:", 'green', attrs=['bold']),
              f"mass={cur_mass:.4f}  friction={cur_friction:.4f}  radius={cur_size:.5f}")

        # Run a handful of random steps to make sure nothing explodes
        for _ in range(steps_per_episode):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    # ---- Sanity checks ----
    print(colored("\n--- Sanity checks ---", 'yellow', attrs=['bold']))
    all_same_mass = len(set(round(m, 8) for m in masses)) == 1
    all_same_friction = len(set(round(f, 8) for f in frictions)) == 1
    all_same_size = len(set(round(s, 8) for s in sizes)) == 1

    if all_same_mass:
        print(colored("WARNING: mass was identical across all episodes!", 'red'))
    else:
        print(colored("PASS: mass varies across episodes", 'green'))

    if all_same_friction:
        print(colored("WARNING: friction was identical across all episodes!", 'red'))
    else:
        print(colored("PASS: friction varies across episodes", 'green'))

    if all_same_size:
        print(colored("WARNING: size was identical across all episodes!", 'red'))
    else:
        print(colored("PASS: size varies across episodes", 'green'))

    # Verify values fall within configured ranges
    in_range = True
    for m in masses:
        if not (0.5 <= m <= 1.5):
            in_range = False
            print(colored(f"FAIL: mass {m} outside range [0.5, 1.5]", 'red'))
    for f in frictions:
        if not (0.5 <= f <= 1.5):
            in_range = False
            print(colored(f"FAIL: friction {f} outside range [0.5, 1.5]", 'red'))
    for s in sizes:
        if not (0.015 <= s <= 0.03):
            in_range = False
            print(colored(f"FAIL: size {s} outside range [0.015, 0.03]", 'red'))
    if in_range:
        print(colored("PASS: all values within configured ranges", 'green'))

    env.close()
    print(colored("\n===== Dry-run complete! =====\n", 'cyan', attrs=['bold']))


if __name__ == '__main__':
    main()
