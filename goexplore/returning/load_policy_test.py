import gym

import unittest
from unittest import mock

from goexplore.envs import save_loadable
from goexplore.archive import loadable_cell
from goexplore.archive import cell
from goexplore.returning import load_policy


class LoadPolicyTest(unittest.TestCase):
    def test_gym_loads_cell_using_saved_snapshot(self):
        save_function = mock.MagicMock()
        load_function = mock.MagicMock()

        core_env = gym.Env()
        env = save_loadable.SaveLoadableWrapper(
            core_env, save_function, load_function
        )

        cell = loadable_cell.LoadableCell(
            snapshot_data="snapshot_data"
        )

        policy = load_policy.LoadPolicy(env)
        policy.return_to_cell(cell)

        load_function.assert_called_once_with(
            core_env, "snapshot_data"
        )

    def test_load_cell_raises_exception_on_non_loadable_env(self):
        class EnvCore(gym.Env):
            def __repr__(self):
                return "REPRENV"
        core_env = EnvCore()
        with self.assertRaisesRegexp(
                TypeError, 'SaveLoadableEnv.*got.*REPRENV'):
            load_policy.LoadPolicy(core_env)

    def test_load_cell_raises_exception_on_non_loadable_cell(self):
        save_function = mock.MagicMock()
        load_function = mock.MagicMock()

        core_env = gym.Env()
        env = save_loadable.SaveLoadableWrapper(
            core_env, save_function, load_function
        )

        test_cell = cell.Cell()
        policy = load_policy.LoadPolicy(env)
        with self.assertRaisesRegexp(TypeError, "LoadableCell.*got.*Cell"):
            policy.return_to_cell(test_cell)





if __name__ == "__main__":
    unittest.main()
