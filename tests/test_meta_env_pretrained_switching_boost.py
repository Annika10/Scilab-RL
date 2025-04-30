import unittest
import gymnasium as gym
from src.custom_envs.register_envs import register_custom_envs
from src.custom_envs.moonlander.meta_env_pretrained_with_switching_boost import SwitchingBoostWrapperEnv


class TestMetaEnvPretrainedSwitchingBoost(unittest.TestCase):
    def test_boosting(self) -> None:
        # register custom envs
        register_custom_envs()
        # define env
        env = gym.make("MetaEnv-pretrained-without-SoC-hard-hard-v0", render_mode="human",
                       # standard path '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/custom_envs/moonlander'
                       config_file_name_collect_task_one="../../../tests/test_data/levels/config_collect_task_one_hard.yaml",
                       config_file_name_collect_task_two="../../../tests/test_data/levels/config_collect_task_two_hard.yaml",
                       input_noise_in_subtasks_on=True,
                       consecutive_frames=5,
                       normalize_rewards=True
                       )
        env = SwitchingBoostWrapperEnv(env=env, boost_value=0.05)
        
        env.reset()
        with self.subTest("no switch"):
            _, reward, _, _, info = env.step(0)
            self.assertEqual(info["reward_without_boost"], reward)
        with self.subTest("again no switch"):
            _, reward, _, _, info = env.step(0)
            self.assertEqual(info["reward_without_boost"], reward)
        with self.subTest("switch"):
            _, reward, _, _, info = env.step(1)
            self.assertEqual(info["reward_without_boost"], reward - 0.05)
        with self.subTest("again switch"):
            _, reward, _, _, info = env.step(0)
            self.assertEqual(info["reward_without_boost"], reward - 0.05)


if __name__ == '__main__':
    unittest.main()
