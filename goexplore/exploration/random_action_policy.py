import gym

from goexplore.exploration import base_policy


class RandomActionPolicy(base_policy.BasePolicy):
    """An exploration policy that samples random action"""

    def __init__(self, environment: gym.Env, number_of_steps: int):
        """Create random exploration policy.

        :param environment: Environment to explore
        :param number_of_steps: Number of steps in an exploration episode
        """
        super().__init__(environment)
        self._number_of_steps = number_of_steps

    def explore(self, current_state: gym.Space):
        """Explore randomly from current_state

        If the environment returns "is_done" flag, then exploration stops.

        :param current_state: State to explore from
        :return: Latest tuple from exploration
        """
        result_tuple = None
        for step_no in range(self._number_of_steps):
            action = self.environment.action_space.sample()
            result_tuple = self._environment_act(current_state, action)
            current_state, _, is_done, _ = result_tuple
            if is_done:
                break
        return result_tuple
