import gym

from goexplore.exploration import base_policy

class RandomActionPolicy(base_policy.BasePolicy):
    """An exploration policy that samples random action"""
    def __init__(self,
                 environment: gym.Env,
                 number_of_steps: int):
        super().__init__(environment)
        self._number_of_steps = number_of_steps

    def explore(self,
                current_state: gym.Space):
        result_tuple = None
        for step_no in range(self._number_of_steps):
            action = self.environment.action_space.sample()
            result_tuple = self._environment_act(current_state, action)
            current_state, _, is_done, _ = result_tuple
            if is_done:
                break
        return result_tuple

