import copy
from logging import getLogger
import gym
import logging
from collections import OrderedDict
import numpy as np

"""Code heavily based on MineRL baselines repo (https://github.com/minerllabs/baselines) with some slight modifications to match stable-baselines"""


class ObtainPoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces['pov']

    def observation(self, observation):
        return observation['pov']


class PoVWithCompassAngleWrapper(gym.ObservationWrapper):
    """Take 'pov' value (current game display) and concatenate compass angle information with it, as a new channel of image;
    resulting image has RGB+compass (or K+compass for gray-scaled image) channels.
    """

    def __init__(self, env):
        super().__init__(env)

        self._compass_angle_scale = 180 / 255  # NOTE: `ScaledFloatFrame` will scale the pixel values with 255.0 later

        pov_space = self.env.observation_space.spaces['pov']
        compass_angle_space = self.env.observation_space.spaces['compassAngle']

        low = self.observation({'pov': pov_space.low, 'compassAngle': compass_angle_space.low})
        high = self.observation({'pov': pov_space.high, 'compassAngle': compass_angle_space.high})

        self.observation_space = gym.spaces.Box(low=low, high=high)


class SerialDiscreteActionWrapper(gym.ActionWrapper):
    """Convert MineRL env's `Dict` action space as a serial discrete action space.

    The term "serial" means that this wrapper can only push one key at each step.
    "attack" action will be alwarys triggered.

    Parameters
    ----------
    env
        Wrapping gym environment.
    always_keys
        List of action keys, which should be always pressed throughout interaction with environment.
        If specified, the "noop" action is also affected.
    reverse_keys
        List of action keys, which should be always pressed but can be turn off via action.
        If specified, the "noop" action is also affected.
    exclude_keys
        List of action keys, which should be ignored for discretizing action space.
    exclude_noop
        The "noop" will be excluded from discrete action list.
    num_camera_discretize
        Number of discretization of yaw control (must be odd).
    allow_pitch
        If specified, this wrapper appends commands to control pitch.
    max_camera_range
        Maximum value of yaw control.
    """

    BINARY_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']

    def __init__(self, env, always_keys=None, reverse_keys=None, exclude_keys=None, exclude_noop=False,
                 num_camera_discretize=3, allow_pitch=False,
                 max_camera_range=10):
        super().__init__(env)

        self.always_keys = [] if always_keys is None else always_keys
        self.reverse_keys = [] if reverse_keys is None else reverse_keys
        self.exclude_keys = [] if exclude_keys is None else exclude_keys
        if len(set(self.always_keys) | set(self.reverse_keys) | set(self.exclude_keys)) != \
                len(self.always_keys) + len(self.reverse_keys) + len(self.exclude_keys):
            raise ValueError('always_keys ({}) or reverse_keys ({}) or exclude_keys ({}) intersect each other.'.format(
                self.always_keys, self.reverse_keys, self.exclude_keys))
        self.exclude_noop = exclude_noop

        self.wrapping_action_space = self.env.action_space
        self.num_camera_discretize = num_camera_discretize
        self._noop_template = OrderedDict([
            ('forward', 0),
            ('back', 0),
            ('left', 0),
            ('right', 0),
            ('jump', 0),
            ('sneak', 0),
            ('sprint', 0),
            ('attack', 0),
            ('camera', np.zeros((2,), dtype=np.float32)),
            # 'none', 'dirt' (Obtain*:)+ 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'
            ('place', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'
            ('equip', 0),
            # (Obtain* tasks only) 'none', 'torch', 'stick', 'planks', 'crafting_table'
            ('craft', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'
            ('nearbyCraft', 0),
            # (Obtain* tasks only) 'none', 'iron_ingot', 'coal'
            ('nearbySmelt', 0),
            ('ESC', 0),
            ('drop', 0),
            ('hotbar.1', 0),
            ('hotbar.2', 0),
            ('hotbar.3', 0),
            ('hotbar.4', 0),
            ('hotbar.5', 0),
            ('hotbar.6', 0),
            ('hotbar.7', 0),
            ('hotbar.8', 0),
            ('hotbar.9', 0),
            ('hotbar.0', 0),
            ('inventory', 0),
        ])
        for key, space in self.wrapping_action_space.spaces.items():
            if key not in self._noop_template:
                raise ValueError('Unknown action name: {}'.format(key))

        # get noop
        self.noop = copy.deepcopy(self._noop_template)
        for key in self._noop_template:
            if key not in self.wrapping_action_space.spaces:
                del self.noop[key]

        # check&set always_keys
        for key in self.always_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `always_keys`.'.format(key))
            self.noop[key] = 1
        #logger.info('always pressing keys: {}'.format(self.always_keys))
        # check&set reverse_keys
        for key in self.reverse_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `reverse_keys`.'.format(key))
            self.noop[key] = 1
        #logger.info('reversed pressing keys: {}'.format(self.reverse_keys))
        # check exclude_keys
        for key in self.exclude_keys:
            if key not in self.noop:
                raise ValueError('unknown exclude_keys: {}'.format(key))
        #logger.info('always ignored keys: {}'.format(self.exclude_keys))

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key in self.always_keys or key in self.exclude_keys:
                continue
            if key in self.BINARY_KEYS:
                # action candidate : {1}  (0 is ignored because it is for noop), or {0} when `reverse_keys`.
                op = copy.deepcopy(self.noop)
                if key in self.reverse_keys:
                    op[key] = 0
                else:
                    op[key] = 1
                self._actions.append(op)
            elif key == 'camera':
                # action candidate : {[0, -max_camera_range], [0, -max_camera_range + delta_range], ..., [0, max_camera_range]}
                # ([0, 0] is excluded)
                delta_range = max_camera_range * 2 / (self.num_camera_discretize - 1)
                if self.num_camera_discretize % 2 == 0:
                    raise ValueError('Number of camera discretization must be odd.')
                for i in range(self.num_camera_discretize):
                    op = copy.deepcopy(self.noop)
                    if i < self.num_camera_discretize // 2:
                        op[key] = np.array([0, -max_camera_range + delta_range * i], dtype=np.float32)
                    elif i > self.num_camera_discretize // 2:
                        op[key] = np.array([0, -max_camera_range + delta_range * (i - 1)], dtype=np.float32)
                    else:
                        continue
                    self._actions.append(op)

                if allow_pitch:
                    for i in range(self.num_camera_discretize):
                        op = copy.deepcopy(self.noop)
                        if i < self.num_camera_discretize // 2:
                            op[key] = np.array([-max_camera_range + delta_range * i, 0], dtype=np.float32)
                        elif i > self.num_camera_discretize // 2:
                            op[key] = np.array([-max_camera_range + delta_range * (i - 1), 0], dtype=np.float32)
                        else:
                            continue
                        self._actions.append(op)

            elif key in {'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt'}:
                # action candidate : {1, 2, ..., len(space)-1}  (0 is ignored because it is for noop)
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)
        if self.exclude_noop:
            del self._actions[0]

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        #logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        #logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action