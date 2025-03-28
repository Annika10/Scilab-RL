import unittest
import os
import yaml
from unittest import mock
import numpy as np

from src.custom_envs.moonlander.helper_functions import read_test_data
from src.custom_envs.moonlander.meta_env_pretrained import MetaEnvPretrained
from src.custom_envs.register_envs import register_custom_test_envs
from gymnasium.error import NameNotFound


class TestMetaEnvPretrained(unittest.TestCase):

    def test_loading_agents_fails(self) -> None:
        with self.subTest("raise FileNotFoundError if the model with the given name does not exist"):
            with self.assertRaises(FileNotFoundError):
                env = MetaEnvPretrained("dodge_best_fm_23_09_rl_model_best", "collect_best_fm_23_08_rl_model_best")
        with self.subTest("raise FileNotFoundError if the id of the given model is not registered"):
            register_custom_test_envs()
            with self.assertRaises(RuntimeError):
                env = MetaEnvPretrained("dodge_pos_neg_rl_model_best", "collect_best_fm_23_08_rl_model_best")
        with self.subTest("raise NameNotFound error, if the configs are not registered"):
            register_custom_test_envs()
            with self.assertRaises(NameNotFound):
                env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best","collect_best_fm_23_08_rl_model_best",
                                        config_file_name_dodge_asteroids='dodge_test_config.yaml', config_file_name_collect_asteroids='collect_test_config.yaml')


    def test_loading_configs(self) -> None:
        config_path_dodge_asteroids = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   "../src/custom_envs/moonlander/standard_config.yaml")
        with open(config_path_dodge_asteroids, "r") as file:
            config_dodge_asteroids = yaml.safe_load(file)

        dodge_difficulty = config_dodge_asteroids["world"]["difficulty"]
        self.assertEqual(dodge_difficulty, 'hard')

    def test_init_raised_Value_Error(self) -> None:
        register_custom_test_envs()
        with self.subTest(
                "raise Value error, if SoC is not in observation, but the observation should only consist of the SoC"):
            with self.assertRaises(ValueError):
                env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best", "collect_best_fm_23_08_rl_model_best", None,
                                        None, None, True, False, obs_is_SoC=True)
        with self.subTest(
                "raise Value error, when two reward functions are chosen (reward good switch decision & reward function of paper"):
            with self.assertRaises(ValueError):
                env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best", "collect_best_fm_23_08_rl_model_best", None,
                                    None, None, False, False, False,
                                        False, False, False, True, True)

    def test_init_everything_set_to_their_initial_value(self) -> None:
        register_custom_test_envs()
        env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best", "collect_best_fm_23_08_rl_model_best")
        self.assertEqual(env.SoC_dodge, np.array([1.0]))
        self.assertEqual(env.SoC_collect, np.array([1.0]))
        self.assertEqual(env.prediction_error_dodge, -1)
        self.assertEqual(env.prediction_error_collect, -1)
        self.assertEqual(env.need_for_control_dodge, -1)
        self.assertEqual(env.need_for_control_collect, -1)

    def test_step_not_one_or_zero(self) -> None:
        register_custom_test_envs()
        env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best", "collect_best_fm_23_08_rl_model_best")
        with self.subTest(
                "raise Value error, when the action is not 0 or 1"):
            with self.assertRaises(ValueError):
                env.actual_step_logic(action=3, task_switch=False)


    def test_environment_task_switch(self) -> None:
        register_custom_test_envs()
        environment = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best", "collect_best_fm_23_08_rl_model_best")

        with self.subTest("test if actions are registered in info dict"):

            test_data = read_test_data("test_meta_env")

            #np.testing.assert_array_equal(
            #    np.array(test_data[0]).flatten(),
            #    environment.state["image"],
            #)

            # first step
            # avoid task
            state, reward, is_done, truncated, info = environment.step(action=0)
            self.assertEqual(info["action_meta"], 0)
            self.assertFalse(is_done)

            # task switch
            # collect task
            state, reward, is_done, truncated, info = environment.step(action=1)
            self.assertEqual(info["action_meta"], 1)
            self.assertFalse(is_done)

if __name__ == '__main__':
    unittest.main()