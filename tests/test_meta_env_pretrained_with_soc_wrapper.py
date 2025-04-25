import unittest
from unittest import mock
import gymnasium as gym
import numpy as np
import math
from src.custom_envs.moonlander.meta_env_pretrained_with_soc_wrapper import SoCWrapperEnv
from src.custom_envs.register_envs import register_custom_envs


class TestMetaEnvPretrainedWithSoC(unittest.TestCase):
    
    @mock.patch("src.custom_algorithms.ppo_moonlander.ppo_moonlander.PPO_MOONLANDER.predict")
    def test_empty_situation(self, ppo_moonlander_predict_mock) -> None:
        # internally we do 5 predictions + 30 in NfC
        ppo_moonlander_predict_mock.side_effect = ([(np.array([2]), None)] * 35)
        # register custom envs
        register_custom_envs()
        # define env
        # reset in this test & in meta_env_pretrained_without_soc
        env = gym.make("MetaEnv-pretrained-without-SoC-hard-hard-v0", render_mode="human",
                       # standard path '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/custom_envs/moonlander'
                       config_file_name_collect_task_one="../../../tests/test_data/levels/config_collect_task_one_hard.yaml",
                       config_file_name_collect_task_two="../../../tests/test_data/levels/config_collect_task_two_hard.yaml",
                       input_noise_in_subtasks_on=False, consecutive_frames=5
                       )
        env = SoCWrapperEnv(env=env)
        env.reset()
        
        # initial x position 15 for task 0 and 25 for task 1
        
        # starting active state
        # [[15  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]
        # expected new state (5 frames)
        # [[25  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]
        
        # starting inactive state
        # [[25  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]
        # expected new state (5 frames)
        # [[25  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]
        
        state, reward, is_done, _, info = env.step(0)
        # active SoC
        self.assertEqual(state[0], 1)
        # inactive SoC
        self.assertEqual(state[1], 1.0 - (1 / 30))
        
        # evaluate starting SoCs (1.0 and 1.0)
        self.assertEqual(reward, 0)
        
        # evaluate prediction error and need for control
        self.assertEqual(info["prediction_error"], 0)
        self.assertEqual(info["need_for_control"], 0)
    
    @mock.patch("src.custom_algorithms.ppo_moonlander.ppo_moonlander.PPO_MOONLANDER.predict")
    def test_different_situations_without_input_noise(self, ppo_moonlander_predict_mock) -> None:
        # internally we do 5 predictions + 30 in NfC
        ppo_moonlander_predict_mock.side_effect = ([(np.array([2]), None)] * 35
                                                   + [(np.array([2]), None)] * 35
                                                   + [(np.array([2]), None)] * 35
                                                   + [(np.array([2]), None)] * 35)
        # register custom envs
        register_custom_envs()
        # define env
        # reset in this test & in meta_env_pretrained_without_soc
        env = gym.make("MetaEnv-pretrained-without-SoC-hard-hard-v0", render_mode="human",
                       # standard path '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/custom_envs/moonlander'
                       config_file_name_collect_task_one="../../../tests/test_data/levels/config_collect_task_one_hard.yaml",
                       config_file_name_collect_task_two="../../../tests/test_data/levels/config_collect_task_two_hard.yaml",
                       input_noise_in_subtasks_on=False, consecutive_frames=5,
                       list_of_object_dict_lists_collect_task_one_filename="../../../tests/test_data/collect_0_self_defined.csv",
                       list_of_object_dict_lists_collect_task_two_filename="../../../tests/test_data/collect_1_self_defined.csv",
                       )
        env = SoCWrapperEnv(env=env)
        env.reset()
        
        # initial x position 15 for task 0 and 25 for task 1
        
        with self.subTest("evaluate starting SoCs, no prediction error = no input noise, NfC = 0"):
            # starting active state
            # [[15  1 17  5 35  5 23  6 28 10 20 14 25 21 35 23 39 28  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames) (17, 5) & (23, 1) is collected
            # [[25  1 35  0 28  5 20  9 25 16 35 18 39 23 26 28  0  0  0  0  0  0]]
            
            # starting inactive state
            # [[25  1 30  4 27  9 16 15 12 20 24 26  7 29  0  0  0  0  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames) (no incoming objects)
            # [[25  1 30 -1 27  4 16 10 12 15 24 21  7 24  0  0  0  0  0  0  0  0]]
            
            state, reward, is_done, _, info = env.step(0)
            # active SoC (PE = 0, NfC = 0)
            self.assertEqual(state[0], 1)
            # inactive SoC
            self.assertEqual(state[1], 1.0 - (1 / 30))
            
            # evaluate starting SoCs (1.0 and 1.0)
            self.assertEqual(reward, 0)
            
            # evaluate prediction error and need for control
            self.assertEqual(info["prediction_error"], 0)
            self.assertEqual(info["need_for_control"], 0)
        
        with self.subTest("evaluate first SoCs, no prediction error = no input noise, NfC = 0"):
            # starting active state
            # [[25  1 35  0 28  5 20  9 25 16 35 18 39 23 26 28  0  0  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames) (28, 5) is collected
            # [[35  1 20  4 25 11 35 13 39 18 26 23 25 29  0  0  0  0  0  0  0  0]]
            
            # starting inactive state
            # [[25  1 30 -1 27  4 16 10 12 15 24 21  7 24  0  0  0  0  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames) (no incoming objects)
            # [[25  1 27 -1 16  5 12 10 24 16  7 19  0  0  0  0  0  0  0  0  0  0]]
            
            state, reward, is_done, _, info = env.step(0)
            # active SoC (PE = 0, NfC = 0)
            self.assertEqual(state[0], 1)
            # inactive SoC
            self.assertEqual(state[1], (1.0 - (1 / 30)) - (1 / 30))
            
            # evaluate last SoCs, task one: 1, task two: 0.97 -> should have switched --> reward of -1
            self.assertEqual(reward, -1)
            
            # evaluate prediction error and need for control
            self.assertEqual(info["prediction_error"], 0)
            # zero because default policy collects more than optimal policy (optimal is faked by me)
            self.assertEqual(info["need_for_control"], 0)
        
        with self.subTest(
                "evaluate second SoCs, no prediction error = no input noise, NfC = 0 -> should have switched"):
            # starting active state
            # [[35  1 20  4 25 11 35 13 39 18 26 23 25 29  0  0  0  0  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames)
            # [[39  1 20 -1 25  6 35  8 39 13 26 18 25 24 39 29  0  0  0  0  0  0]]
            
            # starting inactive state
            # [[25  1 27 -1 16  5 12 10 24 16  7 19  0  0  0  0  0  0  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames) (no incoming objects)
            # [[25  1 16  0 12  5 24 11  7 14  0  0  0  0  0  0  0  0  0  0  0  0]]
            
            state, reward, is_done, _, info = env.step(0)
            # active SoC (PE = 0, NfC = 0)
            self.assertEqual(state[0], 1)
            # inactive SoC
            self.assertEqual(state[1], ((1.0 - (1 / 30)) - (1 / 30)) - (1 / 30))
            
            # evaluate last SoCs, task one: 1, task two: 0.93 -> take action zero is bad --> reward of -1
            self.assertEqual(reward, -1)
            
            # evaluate prediction error and need for control
            self.assertEqual(info["prediction_error"], 0)
            # zero because default policy collects more than optimal policy (optimal is faked by me)
            self.assertEqual(info["need_for_control"], 0)
        
        with self.subTest("switch, evaluate third SoCs, no prediction error = no input noise, NfC = 0"):
            # starting active state
            # [[25  1 16  0 12  5 24 11  7 14  0  0  0  0  0  0  0  0  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames)
            # [[35  1 12  0 24  6  7  9 16 14 16 17 17 25 17 28  0  0  0  0  0  0]]
            
            # starting inactive state
            # [[39  1 20 -1 25  6 35  8 39 13 26 18 25 24 39 29  0  0  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames) (no incoming objects)
            # [[39  1 25  1 35  3 39  8 26 13 25 19 39 24  0  0  0  0  0  0  0  0]]
            
            state, reward, is_done, _, info = env.step(1)
            # inactive SoC
            self.assertEqual(state[0], 1.0 - (1 / 30))
            # active SoC, (PE = 0, NfC = 0)
            self.assertEqual(state[1], 1)
            
            # evaluate last SoCs, task one: 1, task two: 0.9 -> take action one is good --> reward of 1
            self.assertEqual(reward, 1)
            
            # evaluate prediction error and need for control
            self.assertEqual(info["prediction_error"], 0)
            # zero because default policy collects more than optimal policy (optimal is faked by me)
            self.assertEqual(info["need_for_control"], 0)
    
    @mock.patch("src.custom_algorithms.ppo_moonlander.ppo_moonlander.PPO_MOONLANDER.predict")
    @mock.patch("src.custom_envs.moonlander.moonlander_env.MoonlanderWorldEnv.get_input_noise")
    def test_different_situations_with_input_noise(self, input_noise_mock, ppo_moonlander_predict_mock) -> None:
        # input noise for 5 frames per step (4)
        input_noise_mock.side_effect = [1, 1, 2, 2, 0,  # first step
                                        1, 0, 1, 0, 0,  # second step
                                        1, 2, -1, -2, 0,
                                        1, 2, -1, -2, 0, ]
        # internally we do 5 predictions + 30 in NfC
        ppo_moonlander_predict_mock.side_effect = ([(np.array([2]), None)] * 35  # first step
                                                   + [(np.array([2]), None)] * 2 + [(np.array([0]), None)] * 3 + [
                                                       (np.array([1]), None)] * 30  # second step
                                                   + [(np.array([2]), None)] * 35
                                                   + [(np.array([2]), None)] * 35)
        # register custom envs
        register_custom_envs()
        # define env
        # reset in this test & in meta_env_pretrained_without_soc
        env = gym.make("MetaEnv-pretrained-without-SoC-hard-hard-v0", render_mode="human",
                       # standard path '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/custom_envs/moonlander'
                       config_file_name_collect_task_one="../../../tests/test_data/levels/config_collect_task_one_hard.yaml",
                       config_file_name_collect_task_two="../../../tests/test_data/levels/config_collect_task_two_hard.yaml",
                       input_noise_in_subtasks_on=True, consecutive_frames=5,
                       list_of_object_dict_lists_collect_task_one_filename="../../../tests/test_data/collect_0_more_objects_self_defined.csv",
                       list_of_object_dict_lists_collect_task_two_filename="../../../tests/test_data/collect_1_self_defined.csv",
                       )
        env = SoCWrapperEnv(env=env)
        env.reset()
        
        # initial x position 15 for task 0 and 25 for task 1
        
        with self.subTest("evaluate starting SoCs, prediction error ~ 1, NfC = 1"):
            # starting active state
            # [[15  1 17  5 35  5 23  6 28 10 20 14 25 21 35 23 39 23 39 28  0  0]]
            # input noise: (1) 15 + 2 + 1 = 18 (2) 18 + 2 + 1 = 21 (3) 21 + 2 + 2 = 25 (4) 25 + 2 + 2 = 29 (5) 29 + 2 + 0 = 31
            # expected new state (with incoming objects) (5 frames) (17, 5) & (23, 1) is collected
            # [[25  1 35  0 28  5 20  9 25 16 35 18 39 18 39 23 26 28  0  0  0  0]]
            # actual new state (with incoming objects) (5 frames) (23, 1) is collected
            # [[31  1 17  0 35  0 28  5 20  9 25 16 35 18 39 18 39 23 26 28  0  0]]
            
            # starting inactive state
            # [[25  1 30  4 27  9 16 15 12 20 24 26  7 29  0  0  0  0  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames) (no incoming objects)
            # [[25  1 30 -1 27  4 16 10 12 15 24 21  7 24  0  0  0  0  0  0  0  0]]
            
            state, reward, is_done, _, info = env.step(0)
            # active SoC (PE ~ 1, NfC = 1)
            self.assertEqual(round(state[0], 2), 0)
            # inactive SoC
            self.assertEqual(state[1], 1.0 - (1 / 30))
            
            # evaluate starting SoCs (1.0 and 1.0)
            self.assertEqual(reward, 0)
            
            # evaluate prediction error and need for control
            self.assertEqual(info["prediction_error"], math.tanh(0.5 * 6))
            self.assertEqual(info["need_for_control"], 1)
        
        with self.subTest("evaluate first SoCs, smaller prediction error, NfC = 0"):
            # starting active state
            # [[31  1 17  0 35  0 28  5 20  9 25 16 35 18 39 18 39 23 26 28  0  0]]
            # input noise: (1) 31 + 2 + 1 = 34 (2) 34 + 2 + 0 = 36 (3) 36 - 2 + 1 = 35 (4) 35 - 2 + 0 = 33 (5) 33 - 2 + 0 = 31
            # expected new state (with incoming objects) (5 frames)
            # [[29  1 28  0 20  4 25 11 35 13 39 13 39 18 26 23 25 29  0  0  0  0  0  0]]
            # actual new state (with incoming objects) (5 frames)
            # [[31  1 28  0 20  4 25 11 35 13 39 13 39 18 26 23 25 29  0  0  0  0  0  0]]
            
            # starting inactive state
            # [[25  1 30 -1 27  4 16 10 12 15 24 21  7 24  0  0  0  0  0  0  0  0]]
            # expected new state (with incoming objects) (5 frames) (no incoming objects)
            # [[25  1 27 -1 16  5 12 10 24 16  7 19  0  0  0  0  0  0  0  0  0  0]]
            
            state, reward, is_done, _, info = env.step(0)
            # active SoC (PE ~0.76, NfC = 0) -> 0.76/2 = 0.38 -> SoC: 1 - 0.38 = 0.62
            self.assertEqual(state[0], 1 - (math.tanh(0.5 * 2) / 2))
            # inactive SoC
            self.assertEqual(state[1], (1.0 - (1 / 30)) - (1 / 30))
            
            # evaluate last SoCs, task one: 0.5, task two: 0.97 -> take action zero is good --> reward of 1
            self.assertEqual(reward, 1)
            
            # evaluate prediction error and need for control
            self.assertEqual(info["prediction_error"], math.tanh(0.5 * 2))
            # zero because default policy = optimal policy (optimal is faked by me)
            self.assertEqual(info["need_for_control"], 0)


if __name__ == '__main__':
    unittest.main()
