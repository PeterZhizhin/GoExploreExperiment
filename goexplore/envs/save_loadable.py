import typing
import gym


class SaveLoadableEnv(gym.Env):
    """OpenAI Gym wrapper that can be save-loaded."""

    def save_snapshot(self):
        """Save current environment state as a snapshot.

        :returning: Some state representation that can be loaded afterwards
        """
        raise NotImplementedError

    def load_snapshot(self, snapshot) -> gym.Space:
        """Set environment to a snapshot state

        :param snapshot: Snapshot that should be loaded
        :returning: Observation of the loaded snapshot
        """
        raise NotImplementedError


class SaveLoadableWrapper(gym.Wrapper, SaveLoadableEnv):
    def __init__(self, env: gym.Env,
                 save_func: typing.Callable[[gym.Env], typing.Any],
                 load_func: typing.Callable[[gym.Env, typing.Any], gym.Space]):
        super().__init__(env)
        self._save_func = save_func
        self._load_func = load_func

    def load_snapshot(self, snapshot):
        return self._load_func(self.env, snapshot)

    def save_snapshot(self):
        return self._save_func(self.env)
