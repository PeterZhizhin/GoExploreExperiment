import typing

import gym

on_state_change_type = typing.Callable[[
    gym.Space,  # State
    gym.Space,  # Action
    float,  # Reward
    gym.Space,  # New state
    bool,  # Is done
    typing.Optional[object],  # Info
],
                                       type(None)]


def _noop_function(*args, **kwargs):
    pass


class BasePolicy(object):
    """A class that represents an exploration policy for Go-Explore"""

    def __init__(self, environment: gym.Env):
        """Create exploration policy.

        :param environment: OpenAI Gym environment that should be explored
        """
        self.environment = environment
        self._on_action = _noop_function

    @property
    def on_action(self):
        return self._on_action

    @on_action.setter
    def on_action(self, new_on_action: on_state_change_type):
        self._on_action = new_on_action

    def _environment_act(self, current_state: gym.Space, action: gym.Space):
        result = self.environment.step(action)
        new_state, reward, done, info = result
        self._on_action(current_state, action, reward, new_state, done, info)
        return result

    def explore(self, current_state: gym.Space):
        """Explore from current state.

        This method should explore from current_state using the exploration
        policy. It can be e.g. random actions, exploration through curiosity,
        etc.

        :param current_state: Current state of the environment
        :returns Latest tuple from env.step call (or None if not explored)
        """
        raise NotImplementedError
