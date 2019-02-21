import unittest
from unittest import mock

import gym

from goexplore.returning.replay_policy import replay_policy
from goexplore.returning.replay_policy import replay_cell_info
from goexplore.cell import base_returning_info


class ReplayPolicyTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_env = mock.create_autospec(
            gym.Env, instance=True, spec_set=True)
        self.return_policy = replay_policy.ReplayPolicy(self.mock_env)

    def test_replay_policy_resets_env_on_empty_trajectory_and_returns_obs(
            self):
        empty_trajectory_cell = replay_cell_info.ReplayCellInfo(trajectory=[])
        self.mock_env.reset.return_value = "reset_observation"
        observation = self.return_policy.return_to_cell(empty_trajectory_cell)

        self.mock_env.reset.assert_called_once()
        self.assertEqual(observation, "reset_observation")

    def test_replay_policy_replays_whole_policy(self):
        trajectory_cell = replay_cell_info.ReplayCellInfo(trajectory=[
            "action_1",
            "action_2",
            "action_3",
            "action_4",
        ])
        self.mock_env.reset.return_value = "obs_0"
        self.mock_env.step.side_effect = [
            ("obs_1", 100, False, None),
            ("obs_2", 200, False, None),
            ("obs_3", 300, False, None),
            ("obs_4", 400, False, None),
        ]

        obs = self.return_policy.return_to_cell(trajectory_cell)
        self.assertEqual(obs, "obs_4")

        self.mock_env.reset.assert_called_once()
        self.mock_env.step.assert_has_calls([
            mock.call("action_1"),
            mock.call("action_2"),
            mock.call("action_3"),
            mock.call("action_4"),
        ])

    def test_calling_with_unexpected_cell_type_raises_exception(self):
        test_cell = base_returning_info.BaseReturningInfo()
        policy = replay_policy.ReplayPolicy(self.mock_env)
        with self.assertRaisesRegexp(TypeError,
                                     "ReplayCellInfo.*got.*BaseReturningInfo"):
            policy.return_to_cell(test_cell)


if __name__ == '__main__':
    unittest.main()
