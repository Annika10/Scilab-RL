import unittest
from src.custom_envs.moonlander.helper_functions import calculate_gaussian_with_distance_reward


class TestGaussianWithDistanceRewardFunction(unittest.TestCase):
    def test_no_objects_or_not_reachable_collect(self):
        with self.subTest("action 0"):
            with self.subTest("no objects"):
                self.assertEqual(-0.1,
                                 calculate_gaussian_with_distance_reward(x_position_of_agent=5, y_position_of_agent=10,
                                                                         following_observation_size=30, agent_size=2,
                                                                         task="collect", action=0, reward_gaussian=0,
                                                                         object_dict_list=[]))
            with self.subTest("not reachable"):
                self.assertEqual(-0.1,
                                 calculate_gaussian_with_distance_reward(x_position_of_agent=15, y_position_of_agent=1,
                                                                         following_observation_size=30, agent_size=2,
                                                                         task="collect", action=0, reward_gaussian=0,
                                                                         object_dict_list=[{"x": 4, "y": 3, "size": 2}]
                                                                         ))
        with self.subTest("action 1"):
            with self.subTest("no objects"):
                self.assertEqual(0,
                                 calculate_gaussian_with_distance_reward(x_position_of_agent=5, y_position_of_agent=10,
                                                                         following_observation_size=30, agent_size=2,
                                                                         task="collect", action=1, reward_gaussian=0,
                                                                         object_dict_list=[]))
            with self.subTest("not reachable"):
                self.assertEqual(0,
                                 calculate_gaussian_with_distance_reward(x_position_of_agent=15, y_position_of_agent=1,
                                                                         following_observation_size=30, agent_size=2,
                                                                         task="collect", action=1, reward_gaussian=0,
                                                                         object_dict_list=[{"x": 4, "y": 3, "size": 2}]
                                                                         ))
        with self.subTest("action 2"):
            with self.subTest("no objects"):
                self.assertEqual(-0.1,
                                 calculate_gaussian_with_distance_reward(x_position_of_agent=5, y_position_of_agent=10,
                                                                         following_observation_size=30, agent_size=2,
                                                                         task="collect", action=2, reward_gaussian=0,
                                                                         object_dict_list=[]))
            with self.subTest("not reachable"):
                self.assertEqual(-0.1,
                                 calculate_gaussian_with_distance_reward(x_position_of_agent=15, y_position_of_agent=1,
                                                                         following_observation_size=30, agent_size=2,
                                                                         task="collect", action=2, reward_gaussian=0,
                                                                         object_dict_list=[{"x": 4, "y": 3, "size": 2}]
                                                                         ))


if __name__ == '__main__':
    unittest.main()
