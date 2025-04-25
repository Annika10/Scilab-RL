import unittest
from unittest import mock
import gymnasium as gym
import numpy as np
from src.custom_envs.register_envs import register_custom_envs


class TestMetaEnvPretrainedWithoutSoC(unittest.TestCase):
    @mock.patch("src.custom_algorithms.ppo_moonlander.ppo_moonlander.PPO_MOONLANDER.predict")
    @mock.patch("src.custom_envs.moonlander.moonlander_env.MoonlanderWorldEnv.get_input_noise")
    def test_different_situations(self, input_noise_mock, ppo_moonlander_predict_mock) -> None:
        # with input noise, to get different belief to actual state
        # mock input noise
        input_noise_mock.side_effect = [1, 2, -1, -2, 0, -3, 3, 1]
        # we do 8 steps
        ppo_moonlander_predict_mock.side_effect = ([(np.array([2]), None)] * 8)
        
        # register custom envs
        register_custom_envs()
        # define env
        env = gym.make("MetaEnv-pretrained-without-SoC-hard-hard-v0", render_mode="human",
                       # standard path '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/custom_envs/moonlander'
                       config_file_name_collect_task_one="../../../tests/test_data/levels/config_collect_task_one_hard.yaml",
                       config_file_name_collect_task_two="../../../tests/test_data/levels/config_collect_task_two_hard.yaml",
                       input_noise_in_subtasks_on=True,
                       list_of_object_dict_lists_collect_task_one_filename="../../../tests/test_data/collect_0_different_situations_self_defined.csv",
                       list_of_object_dict_lists_collect_task_two_filename="../../../tests/test_data/collect_1_self_defined.csv",
                       )
        
        env.reset()
        with self.subTest("three steps action 0"):
            state, reward, is_done, _, info = env.step(0)
            
            ### active task
            # actual state
            # agent: initial x position: 15, go to the right --> 17 + input noise of 1 --> 18
            # all objects: y position -1 (already before -1 in init & reset, I don't know why)
            new_state_active = np.array([18, 1, 13, 1, 35, 4, 28, 9, 20, 13, 25, 20, 35, 22, 39, 27, 0, 0, 0, 0, 0, 0])
            # we cannot see the belief state of the active task
            # first 22 elements because of action 0
            np.testing.assert_array_equal(new_state_active, state[:22])
            
            ### inactive task belief state:
            # agent: initial x position: 25, go one step down --> 25 without input noise, no new incoming objects
            # all objects: y position -1 (already before -1 in init & reset, I don't know why)
            new_state_inactive = np.array([25, 1, 30, 3, 27, 8, 16, 14, 12, 19, 24, 25, 7, 28, 0, 0, 0, 0, 0, 0, 0, 0])
            # we cannot see the actual state of the inactive task
            # from 22 to 44 elements because of action 0
            np.testing.assert_array_equal(new_state_inactive, state[22:])
            
            state, reward, is_done, _, info = env.step(0)
            
            ### active task
            # actual state
            # agent: x position: 18, go to the right --> 20 + input noise of 2 --> 22
            # all objects: y position -1 + new incoming objects
            new_state_active = np.array([22, 1, 13, 0, 35, 3, 28, 8, 20, 12, 25, 19, 35, 21, 39, 26, 0, 0, 0, 0, 0, 0])
            # we cannot see the belief state of the active task
            # first 22 elements because of action 0
            np.testing.assert_array_equal(new_state_active, state[:22])
            
            ### inactive task belief state:
            # agent: x position: 25, go one step down --> 25 without input noise
            # all objects: y position -1, no new incoming objects
            new_state_inactive = np.array([25, 1, 30, 2, 27, 7, 16, 13, 12, 18, 24, 24, 7, 27, 0, 0, 0, 0, 0, 0, 0, 0])
            # we cannot see the actual state of the inactive task
            # from 22 to 44 elements because of action 0
            np.testing.assert_array_equal(new_state_inactive, state[22:])
            
            state, reward, is_done, _, info = env.step(0)
            
            ### active task
            # actual state
            # agent: x position: 22, go to the right --> 24 + input noise of -1 --> 23
            # all objects: y position -1 + new incoming objects
            new_state_active = np.array(
                [23, 1, 13, -1, 35, 2, 28, 7, 20, 11, 25, 18, 35, 20, 39, 25, 26, 30, 0, 0, 0, 0])
            # we cannot see the belief state of the active task
            # first 22 elements because of action 0
            np.testing.assert_array_equal(new_state_active, state[:22])
            
            ### inactive task belief state:
            # agent: x position: 25, go one step down --> 25 without input noise
            # all objects: y position -1, no new incoming objects
            new_state_inactive = np.array([25, 1, 30, 1, 27, 6, 16, 12, 12, 17, 24, 23, 7, 26, 0, 0, 0, 0, 0, 0, 0, 0])
            # we cannot see the actual state of the inactive task
            # from 22 to 44 elements because of action 0
            np.testing.assert_array_equal(new_state_inactive, state[22:])
        
        with self.subTest("switch to action 1"):
            state, reward, is_done, _, info = env.step(1)
            
            ### active task
            # actual state
            # agent: x position: 25, go to the right --> 27 + input noise of -2 --> 25
            # all objects: y position -1 + new incoming objects
            new_state_active = np.array([25, 1, 30, 0, 27, 5, 16, 11, 12, 16, 24, 22, 7, 25, 16, 30, 0, 0, 0, 0, 0, 0])
            # we cannot see the belief state of the active task
            # from 22 to 44 elements because of action 1
            np.testing.assert_array_equal(new_state_active, state[22:])
            
            ### inactive task belief state:
            # agent: x position: 23, go one step down --> 23 without input noise
            # all objects: y position -1, no new incoming objects
            new_state_inactive = np.array(
                [23, 1, 0, 0, 35, 1, 28, 6, 20, 10, 25, 17, 35, 19, 39, 24, 26, 29, 0, 0, 0, 0])
            # we cannot see the actual state of the inactive task
            # first 22 elements because of action 1
            np.testing.assert_array_equal(new_state_inactive, state[:22])
        
        with self.subTest("three steps action 1"):
            state, reward, is_done, _, info = env.step(1)
            
            ### active task
            # actual state
            # agent: x position: 25, go to the right --> 27 + input noise of 0 --> 27
            # all objects: y position -1 + new incoming objects
            new_state_active = np.array([27, 1, 30, -1, 27, 4, 16, 10, 12, 15, 24, 21, 7, 24, 16, 29, 0, 0, 0, 0, 0, 0])
            # we cannot see the belief state of the active task
            # from 22 to 44 elements because of action 1
            np.testing.assert_array_equal(new_state_active, state[22:])
            
            ### inactive task belief state:
            # agent: x position: 23, go one step down --> 23 without input noise
            # all objects: y position -1, no new incoming objects
            new_state_inactive = np.array(
                [23, 1, 0, 0, 35, 0, 28, 5, 20, 9, 25, 16, 35, 18, 39, 23, 26, 28, 0, 0, 0, 0])
            # we cannot see the actual state of the inactive task
            # first 22 elements because of action 1
            np.testing.assert_array_equal(new_state_inactive, state[:22])
            
            state, reward, is_done, _, info = env.step(1)
            
            ### active task
            # actual state
            # agent: x position: 27, go to the right --> 29 + input noise of -3 --> 26
            # all objects: y position -1 + new incoming objects
            new_state_active = np.array([26, 1, 27, 3, 16, 9, 12, 14, 24, 20, 7, 23, 16, 28, 0, 0, 0, 0, 0, 0, 0, 0])
            # we cannot see the belief state of the active task
            # from 22 to 44 elements because of action 1
            np.testing.assert_array_equal(new_state_active, state[22:])
            
            ### inactive task belief state:
            # agent: x position: 23, go one step down --> 23 without input noise
            # all objects: y position -1, no new incoming objects
            new_state_inactive = np.array(
                [23, 1, 0, 0, 35, -1, 28, 4, 20, 8, 25, 15, 35, 17, 39, 22, 26, 27, 0, 0, 0, 0])
            # we cannot see the actual state of the inactive task
            # first 22 elements because of action 1
            np.testing.assert_array_equal(new_state_inactive, state[:22])
            
            state, reward, is_done, _, info = env.step(1)
            
            ### active task
            # actual state
            # agent: x position: 26, go to the right --> 28 + input noise of 3 --> 31
            # all objects: y position -1 + new incoming objects
            # 27, 3 was collected in the last step
            new_state_active = np.array([31, 1, 16, 8, 12, 13, 24, 19, 7, 22, 16, 27, 16, 28, 0, 0, 0, 0, 0, 0, 0, 0])
            # we cannot see the belief state of the active task
            # from 22 to 44 elements because of action 1
            np.testing.assert_array_equal(new_state_active, state[22:])
            
            ### inactive task belief state:
            # agent: x position: 23, go one step down --> 23 without input noise
            # all objects: y position -1, no new incoming objects
            new_state_inactive = np.array([23, 1, 0, 0, 0, 0, 28, 3, 20, 7, 25, 14, 35, 16, 39, 21, 26, 26, 0, 0, 0, 0])
            # we cannot see the actual state of the inactive task
            # first 22 elements because of action 1
            np.testing.assert_array_equal(new_state_inactive, state[:22])
        
        with self.subTest("switch to action 0"):
            state, reward, is_done, _, info = env.step(0)
            
            ### active task
            # actual state
            # agent: x position: 23, go to the right --> 25 + input noise of 1 --> 26
            # all objects: y position -1 + new incoming objects
            # 27, 3 was collected in the last step
            new_state_active = np.array([26, 1, 28, 2, 20, 6, 25, 13, 35, 15, 39, 20, 26, 25, 0, 0, 0, 0, 0, 0, 0, 0])
            # we cannot see the belief state of the active task
            # first 22 elements because of action 0
            np.testing.assert_array_equal(new_state_active, state[:22])
            
            ### inactive task belief state:
            # agent: x position: 31, go one step down --> 31 without input noise
            # all objects: y position -1, no new incoming objects
            new_state_inactive = np.array([31, 1, 16, 7, 12, 12, 24, 18, 7, 21, 16, 26, 16, 27, 0, 0, 0, 0, 0, 0, 0, 0])
            # we cannot see the actual state of the inactive task
            # from 22 to 44 elements because of action 0
            np.testing.assert_array_equal(new_state_inactive, state[22:])
    
    @mock.patch("src.custom_algorithms.ppo_moonlander.ppo_moonlander.PPO_MOONLANDER.predict")
    def test_consecutive_frames(self, ppo_moonlander_predict_mock) -> None:
        # internally we do 5 predictions
        ppo_moonlander_predict_mock.side_effect = ([(np.array([2]), None)] * 5)
        
        # register custom envs
        register_custom_envs()
        
        env = gym.make("MetaEnv-pretrained-without-SoC-hard-hard-v0", render_mode="human",
                       # standard path '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/custom_envs/moonlander'
                       config_file_name_collect_task_one="../../../tests/test_data/levels/config_collect_task_one_hard.yaml",
                       config_file_name_collect_task_two="../../../tests/test_data/levels/config_collect_task_two_hard.yaml",
                       input_noise_in_subtasks_on=False, consecutive_frames=5,
                       list_of_object_dict_lists_collect_task_one_filename="../../../tests/test_data/collect_0_self_defined.csv",
                       list_of_object_dict_lists_collect_task_two_filename="../../../tests/test_data/collect_1_self_defined.csv",
                       )
        env.reset()
        with self.subTest("action 0 (internally 5 times)"):
            state, reward, is_done, _, info = env.step(0)
            
            ### active task
            # actual state
            # agent: initial x position: 15, go to the right x 5 times --> 17 --> 19 --> 21 --> 23 --> 25
            # all objects: y position -1 x 5 times + new incoming objects (26, 34)
            new_state_active = np.array([25, 1, 35, 0, 28, 5, 20, 9, 25, 16, 35, 18, 39, 23, 26, 28, 0, 0, 0, 0, 0, 0])
            # we cannot see the belief state of the active task
            # first 22 elements because of action 0
            np.testing.assert_array_equal(new_state_active, state[:22])
            
            ### inactive task belief state:
            # agent: initial x position: 25, go one step down --> 25 without input noise
            # all objects: y position -1 x 5 times, no new incoming objects
            new_state_inactive = np.array([25, 1, 30, -1, 27, 4, 16, 10, 12, 15, 24, 21, 7, 24, 0, 0, 0, 0, 0, 0, 0, 0])
            # we cannot see the actual state of the inactive task
            # from 22 to 44 elements because of action 0
            np.testing.assert_array_equal(new_state_inactive, state[22:])
            
            # collect two objects during the 5 steps
            self.assertEqual(info["collect_task_one_collected_objects"], 2)
            self.assertEqual(info["collect_task_two_collected_objects"], 0)
            self.assertListEqual(info["action_of_current_task_agent"], [[2], [2], [2], [2], [2]])


if __name__ == '__main__':
    unittest.main()
