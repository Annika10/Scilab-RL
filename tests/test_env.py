import unittest
from unittest import mock
from tests import ROOT_DIR

import numpy as np
from stable_baselines3.common.env_checker import check_env

from tests.helper_functions_test import load_test_config, read_test_data
from src.custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv


class TestMoonlanderWorldEnvironment(unittest.TestCase):

    def test_environment_with_initial_env(self) -> None:
        with self.subTest("test_environment_with_stable_baselines_initial_env"):
            environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env.yaml")
            check_env(environment)
        with self.subTest("test_empty_environment_with_zero_steps"):
            self.assertRaises(ValueError, MoonlanderWorldEnv,
                              config_file_name=str(ROOT_DIR)+"/test_data/levels/real_empty_env.yaml")


    def test_environment_always_stay(self) -> None:
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env.yaml")

        test_data = read_test_data("test_environment_always_stay")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("first step"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)
        with self.subTest("second step"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)

    def test_environment_always_go_right(self) -> None:
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env.yaml")

        test_data = read_test_data("test_environment_always_go_right")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("first step"):
            state, reward, is_done, truncated, _ = environment.step(action=2)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)
        with self.subTest("second step"):
            state, reward, is_done, truncated, _ = environment.step(action=2)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 0)
            self.assertFalse(is_done)
        #crash
        with self.subTest("third step"):
            state, reward, is_done, truncated, _ = environment.step(action=2)
            expected_observation = test_data[3]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)

    def test_environment_always_go_left(self) -> None:
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env.yaml")

        test_data = read_test_data("test_environment_always_go_left")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("first step"):
            state, reward, is_done, truncated, _ = environment.step(action=0)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)
        with self.subTest("second step"):
            state, reward, is_done, truncated, _ = environment.step(action=0)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 0)
            self.assertFalse(is_done)
        # crash
        with self.subTest("third step"):
            state, reward, is_done, truncated, _ = environment.step(action=0)
            expected_observation = test_data[3]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)

    def test_environment_always_go_left_bigger_size(self) -> None:
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env_bigger_agent.yaml")

        test_data = read_test_data("test_environment_always_go_left_bigger_size")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        # crash
        with self.subTest("first step"):
            state, reward, is_done, truncated, _ = environment.step(action=0)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)


    ### OBSTACLES

    @mock.patch(
        "src.custom_envs.moonlander.helper_functions.create_list_of_object_dicts"
    )
    def test_environment_with_obstacles_and_always_stay(
            self, obstacle_create_mock
    ) -> None:

        obstacle_create_mock.return_value = [
            {"x": 6, "y": 11, "size": 1},
        ]

        # create world with obstacles
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_env_with_obstacles.yaml")

        test_data = read_test_data("test_environment_with_obstacles_and_always_stay")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        # some steps
        with self.subTest("other steps"):
            for i in range(8):
                state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)
        # next step --> at obstacle
        with self.subTest("step at obstacle"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)
        # next step --> pass obstacle
        with self.subTest("step in obstacle"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[3]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)

    @mock.patch(
        "src.custom_envs.moonlander.helper_functions.create_list_of_object_dicts"
    )
    def test_environment_with_obstacles_and_crash(
            self, obstacle_create_mock
    ) -> None:

        obstacle_create_mock.return_value = [
            {"x": 6, "y": 11, "size": 1},
        ]

        # create world with obstacles
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_env_with_obstacles.yaml")

        test_data = read_test_data("test_environment_with_obstacles_and_crash")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        # some steps
        with self.subTest("other steps"):
            for i in range(7):
                state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)
        # next step --> at obstacle
        with self.subTest("step at obstacle"):
            for i in range(2):
                state, reward, is_done, truncated, _ = environment.step(action=0)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 0)
            self.assertFalse(is_done)
        # next step --> crash in obstacle
        with self.subTest("step in obstacle"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[3]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)


    @mock.patch(
        "src.custom_envs.moonlander.helper_functions.create_list_of_object_dicts"
    )
    def test_environment_with_crash_in_obstacle_and_bigger_size(
            self, obstacle_create_mock
    ) -> None:
        obstacle_create_mock.return_value = [
            {"x": 5, "y": 15, "size": 2},
        ]

        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_env_with_obstacles_bigger_agent.yaml")

        test_data = read_test_data("test_environment_with_crash_in_obstacle_and_bigger_size")

        with self.subTest("initial state"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        # tenth step --> at the obstacle
        with self.subTest("some steps --> at the obstacle"):
            for i in range(10):
                state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 0)
            self.assertFalse(is_done)
        # crash in obstacle
        with self.subTest("crash in obstacle"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)

    @mock.patch(
        "src.custom_envs.moonlander.helper_functions.create_list_of_object_dicts"
    )
    def test_environment_obstacle_crash_after_passing(
            self, obstacle_create_mock
    ) -> None:
        obstacle_create_mock.return_value = [
            {"x": 2, "y": 8, "size": 2},
        ]

        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_env_with_obstacles_smaller.yaml")

        test_data = read_test_data("test_environment_obstacle_crash_after_passing")
        with self.subTest("initial state"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("crash after passing"):
            actions = [1, 1, 1, 1, 1, 1, 0, 0]
            for action in actions:
                state, reward, is_done, truncated, _ = environment.step(action=action)

            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)

    @mock.patch(
        "src.custom_envs.moonlander.helper_functions.create_list_of_object_dicts"
    )
    def test_environment_multiple_obstacles_multiple_steps(
            self, obstacle_create_mock
    ) -> None:
        obstacle_create_mock.return_value = [
            {"x": 1, "y": 1, "size": 1},
            {"x": 3, "y": 2, "size": 1},
        ]

        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env.yaml")

        test_data = read_test_data("test_environment_multiple_obstacles_multiple_steps")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("crash"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)

    @mock.patch(
        "src.custom_envs.moonlander.helper_functions.create_list_of_object_dicts"
    )
    def test_environment_collect_coins(self, obstacle_create_mock) -> None:
        obstacle_create_mock.return_value = [
            {"x": 1, "y": 2, "size": 1},
            {"x": 3, "y": 3, "size": 1},
        ]

        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env_collect.yaml")

        test_data = read_test_data("test_environment_collect_coins")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("one step before coin"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 0)
            self.assertFalse(is_done)
        with self.subTest("coin collected"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)

    ### WALLS
    def test_environment_crash_in_funnel(self) -> None:

        # too small for drift
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/easy_env.yaml")

        # the funnels are from 16 to 21 and 28 to 33
        test_data = read_test_data("test_environment_crash_in_funnel")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("step until funnel"):
            for i in range(6):
                state, reward, is_done, truncated, _ = environment.step(action=2)

            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)

    def test_environment_crash_in_inner_funnel(self) -> None:
        # too small for drift
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/easy_env.yaml")

        # drift for getting at suitable position
        drift_ranges = [[1, 40, 1, True, False]]
        environment.drift_ranges_with_drift_number = drift_ranges
        environment.update_observation()

        # the funnels are from 16 to 21 and 28 to 33
        test_data = read_test_data("test_environment_crash_in_inner_funnel")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("step until funnel"):
            for i in range(3):
                state, reward, is_done, truncated, _ = environment.step(action=1)
            for i in range(2):
                state, reward, is_done, truncated, _ = environment.step(action=2)

            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)

    ### DRIFT
    def test_drift(self) -> None:

        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env.yaml")

        # hardcode the drift for better testing
        # otherwise the drift would be defined randomly which is difficult to test
        drift_ranges = [[6, 11, 1, True, False], [12, 15, -1, True, False]]
        environment.drift_ranges_with_drift_number = drift_ranges
        environment.update_observation()

        test_data = read_test_data("test_drift")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("step towards drift"):
            for i in range(4):
                state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)
        with self.subTest("step with drift"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)
        with self.subTest("second step with drift, counter-steering drift"):
            state, reward, is_done, truncated, _ = environment.step(action=0)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)
        with self.subTest("step with drift in other direction"):
            state, reward, is_done, truncated, _ = environment.step(action=0)
            state, reward, is_done, truncated, _ = environment.step(action=0)
            state, reward, is_done, truncated, _ = environment.step(action=0)
            state, reward, is_done, truncated, _ = environment.step(action=1)
            state, reward, is_done, truncated, _ = environment.step(action=0)
            expected_observation = test_data[3]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertFalse(is_done)


    def test_drift_bigger_agent(self) -> None:

        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env_bigger_agent.yaml")

        # hardcode the drift for better testing
        # otherwise the drift would be defined randomly which is difficult to test
        drift_ranges = [[6, 11, 1, True, False], [12, 15, -1, True, False]]
        environment.drift_ranges_with_drift_number = drift_ranges
        environment.update_observation()

        test_data = read_test_data("test_drift_bigger_agent")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("step with drift and crash"):
            state, reward, is_done, truncated, _ = environment.step(action=1)
            state, reward, is_done, truncated, _ = environment.step(action=1)
            state, reward, is_done, truncated, _ = environment.step(action=1)
            state, reward, is_done, truncated, _ = environment.step(action=2)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, -100)
            self.assertTrue(is_done)

    def test_drift_clipping(self) -> None:
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env.yaml")

        # hardcode the drift for better testing
        # otherwise the drift would be defined randomly which is difficult to test
        drift_ranges = [[1, 25, 1, True, False]]
        environment.drift_ranges_with_drift_number = drift_ranges
        environment.update_observation()

        is_done = False

        for i in range(10):
            state, reward, is_done, truncated, _ = environment.step(action=2)
            if is_done:
                break

        self.assertTrue(
            is_done,
            "Agent should have crashed after 10 steps + drift to the right! "
            "Maybe it clipped out of bounds?",
        )

    def test_environment_drift_in_whole_level_ensure_that_one_can_stay_at_one_position(
            self,
    ) -> None:

        # too small for drift
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/whole_level_drift_env.yaml")

        test_data = read_test_data(
            "test_environment_drift_in_whole_level_ensure_that_one_can_stay_at_one_position"
        )

        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            is_done = environment.is_done()
            self.assertFalse(environment.is_done())
        with self.subTest("step until funnel"):
            while not is_done:
                state, reward, is_done, truncated, info = environment.step(action=1)
                if is_done:
                    break
                state, reward, is_done, truncated, info = environment.step(action=2)

            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(reward, 10)
            self.assertTrue(is_done)

    ### INPUT NOISE
    def test_environment_input_noise(self) -> None:

        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env.yaml")

        test_data = read_test_data("test_environment_input_noise")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertFalse(environment.is_done())
        with self.subTest("left step with no input noise"):
            state, _, _, _, _ = environment.step(action=0, step_width=0)
            state, _, _, _, _ = environment.step(action=0, step_width=0)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(environment.x_position_of_agent, 1)
        with self.subTest("right step with high input noise --> stay at same position"):
            state, _, _, _, _ = environment.step(action=2, step_width=-1)
            expected_observation = test_data[1]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(environment.x_position_of_agent, 1)
        with self.subTest("right step with high input noise --> very big step"):
            state, _, _, _, _ = environment.step(action=2, step_width=3)
            expected_observation = test_data[2]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), state
            )
            self.assertEqual(environment.x_position_of_agent, 5)


    def test_environment_step_after_done(self) -> None:
        config = load_test_config("basic_empty_env")
        config["world"].update({"x_width": 5, "y_height": 4})
        config["agent"]["observation_height"] = 3
        environment = MoonlanderWorldEnv(reward_function='simple', config_file_name=str(ROOT_DIR)+"/test_data/levels/basic_empty_env_step_after_done.yaml")

        test_data = read_test_data("test_environment_step_after_done")
        with self.subTest("initialisation"):
            expected_observation = test_data[0]
            np.testing.assert_array_equal(
                np.array(expected_observation).flatten(), environment.state.flatten()
            )
            self.assertTrue(environment.is_done())
        with self.subTest("one step"):
            self.assertRaises(EnvironmentError, environment.step, action=1)


if __name__ == "__main__":
    unittest.main()
