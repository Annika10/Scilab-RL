import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as check_env_sb3
from src.custom_envs.register_envs import register_custom_envs
from src.custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv
from src.custom_envs.moonlander.image_wrapper import ImageWrapperEnv
from src.custom_envs.moonlander.meta_env_pretrained_without_soc import MetaEnvPretrainedWithoutSoC
from src.custom_envs.moonlander.meta_env_pretrained_with_soc_wrapper import SoCRewardOnlyWrapperEnv
from src.custom_envs.moonlander.meta_env_pretrained_with_soc_wrapper import SoCObsAndRewardWrapperEnv
from src.custom_envs.moonlander.meta_env_pretrained_with_soc_observation_only_wrapper import SoCObsOnlyWrapperEnv
from src.custom_envs.moonlander.meta_env_pretrained_with_switching_boost import SwitchingBoostWrapperEnv


def main(environment):
    observation, info = environment.reset()
    # environment.render()
    
    terminated = False
    while not terminated:
        action = (
            environment.action_space.sample()
        )  # agent policy that uses the observation and info*.py
        observation, reward, terminated, truncated, info = environment.step(action=0)
        # environment.render()
    
    environment.close()


if __name__ == "__main__":
    register_custom_envs()
    print("registered custom envs")
    
    env = MetaEnvPretrainedWithoutSoC(
        collect_task_one_best_model_name="collect_gaussian_with_distance_hard_positions_31_03_2025_best_model",
        collect_task_two_best_model_name="collect_gaussian_with_distance_hard_positions_31_03_2025_best_model",
        config_file_name_collect_task_one="config_collect_hard.yaml",
        config_file_name_collect_task_two="config_collect_hard.yaml",
        consecutive_frames=5)
    # env = SwitchingBoostWrapperEnv(env=env, boost_value=0.05)
    env = SoCRewardOnlyWrapperEnv(env=env)
    env = SoCObsAndRewardWrapperEnv(env=env)
    env = SoCObsOnlyWrapperEnv(env=env)
    # env = gym.make("MetaEnv-pretrained-new")
    # env = gym.make("MoonlanderWorld-collect-gaussian_with_distance-hard-v0")
    # model_based_env = ImageWrapperEnv(env=env)
    # env = MetaEnvPretrained(
    #     collect_task_one_best_model_name="collect_gaussian_with_distance_hard_positions_31_03_2025_best_model",
    #     collect_task_two_best_model_name="collect_gaussian_with_distance_hard_positions_31_03_2025_best_model",
    #     config_file_name_collect_task_one="config_collect_hard.yaml",
    #     config_file_name_collect_task_two="config_collect_hard.yaml")
    # check_env(env)
    # check_env_sb3(env)
    main(environment=env)
