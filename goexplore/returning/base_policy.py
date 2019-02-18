import gym
from goexplore.archive import cell


class BasePolicy(object):
    """A class that represents a returning policy for Go-Explore"""

    def __init__(self, environment: gym.Env):
        self._environment = environment

    def return_to_cell(self, expected_cell: cell.Cell):
        """Move the environment to the expected cell.

        :param expected_cell: The cell that the env should be returned to
        :return Observation of the loaded snapshot
        """
        raise NotImplementedError
