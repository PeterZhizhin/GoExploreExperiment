import unittest
from unittest import mock

import gym

from goexplore.envs import save_loadable


class SaveLoadableTest(unittest.TestCase):
    def test_save_loadable_wrapper_uses_passed_functions_for_save_load(self):
        save_func = mock.MagicMock()
        load_func = mock.MagicMock()
        fake_env = mock.create_autospec(gym.Env, instance=True, spec_set=True)

        wrapper = save_loadable.SaveLoadableWrapper(fake_env, save_func,
                                                    load_func)

        wrapper.save_snapshot()
        save_func.assert_called_once_with(fake_env)

        wrapper.load_snapshot('test_snapshot')
        load_func.assert_called_once_with(fake_env, 'test_snapshot')


if __name__ == '__main__':
    unittest.main()
