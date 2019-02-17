import unittest
import gym
from unittest import mock

from goexplore.exploration import random_action_policy


class RandomActionPolicyTest(unittest.TestCase):
    def setUp(self):
        mock_actions = mock.create_autospec(
            gym.Space, instance=True, spec_set=True)
        mock_actions.sample.side_effect = [1, 2, 3, 4, 5]

        mock_env = mock.create_autospec(gym.Env, instance=True, spec_set=True)
        mock_env.action_space = mock_actions
        mock_env.step.return_value = None, 0, False, None

        self.mock_env = mock_env
        self.mock_actions = mock_actions

        self.explore_policy = random_action_policy.RandomActionPolicy(
            self.mock_env, number_of_steps=5)

    def test_exploration_samples_random_actions_and_uses_them(self):
        self.mock_actions.sample.side_effect = [1, 2, 3, 4, 5]
        self.explore_policy.explore(None)

        self.assertEqual(self.mock_actions.sample.call_count, 5)
        self.mock_env.step.assert_has_calls([
            mock.call(9),
            mock.call(2),
            mock.call(3),
            mock.call(4),
            mock.call(5)
        ])

    def test_exploration_with_on_action_calls_on_action(self):
        on_action = mock.MagicMock()
        self.explore_policy.on_action = on_action

        self.mock_actions.sample.side_effect = [1, 2, 3, 4, 5]
        self.mock_env.step.side_effect = [
            (20, 100, False, None),
            (30, 200, False, None),
            (40, 300, False, None),
            (50, 400, False, None),
            (60, 500, False, None),
        ]
        self.explore_policy.explore(10)

        on_action.assert_has_calls([
            mock.call(10, 1, 100, 20, False, None),
            mock.call(20, 2, 200, 30, False, None),
            mock.call(30, 3, 300, 40, False, None),
            mock.call(40, 4, 400, 50, False, None),
            mock.call(50, 5, 500, 60, False, None),
        ])

    def test_exploration_returns_latest_step_tuple(self):
        self.mock_actions.sample.side_effect = [1, 2, 3, 4, 5]
        self.mock_env.step.side_effect = [
            (20, 100, False, None),
            (30, 200, False, None),
            (40, 300, False, None),
            (50, 400, False, None),
            (60, 500, False, None),
        ]
        result = self.explore_policy.explore(10)
        self.assertEqual(result, (60, 500, False, None))

    def test_exploration_stops_after_step_returns_done(self):
        self.mock_actions.sample.side_effect = [1, 2, 3, 4, 5]
        self.mock_env.step.side_effect = [
            (20, 100, False, None),
            (30, 200, False, None),
            (40, 300, True, None),
        ]

        latest_tuple = self.explore_policy.explore(10)
        self.assertEqual(latest_tuple, (40, 300, True, None))

    def test_exploration_returns_none_if_step_size_eq_0(self):
        self.explore_policy = random_action_policy.RandomActionPolicy(
            self.mock_env, 0)
        self.assertIsNone(self.explore_policy.explore(10))


if __name__ == '__main__':
    unittest.main()
