import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
from src.custom_envs.register_envs import register_custom_envs
from src.custom_algorithms.ppo_moonlander.utils import normalize_gaussian_with_distance_reward


def plot_histogram_collect(reward_list: list[float], task_name: str):
    df = pd.Series(reward_list)
    hist = df.hist(bins=70)
    
    # Adding title and labels
    plt.title(f'Histogram for Rewards in {task_name} (n={df.shape[0]})')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    
    # Display the histogram
    plt.show()


def run_through_env(env) -> tuple[list[float], list[float]]:
    reward_list = []
    normalized_reward_list = []
    
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    step_counter = 0
    reward_sum_per_episode = 0
    for _ in range(10000):
        # this is where you would insert your policy
        action = env.action_space.sample()
        
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        # Append the reward to the list
        reward_list.append(reward)
        normalized_reward = normalize_gaussian_with_distance_reward(task="collect", absolute_reward=reward)
        normalized_reward_list.append(normalized_reward)
        
        reward_sum_per_episode += normalized_reward
        step_counter += 1
        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            print(
                f"Reward sum per episode: {reward_sum_per_episode} with {step_counter} steps results in a mean reward of {reward_sum_per_episode / step_counter}")
            step_counter = 0
            reward_sum_per_episode = 0
            observation, info = env.reset()
    
    env.close()
    return reward_list, normalized_reward_list


if __name__ == "__main__":
    # dodge_filename = '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/data/bdf7da1/MoonlanderWorld-dodge-gaussian-v0/15-10-05/dodge_rewards.csv'
    # plot_histogram_dodge(filename=dodge_filename, task_name='Dodge Asteroids')
    
    register_custom_envs()
    
    difficulty = 'hard'
    # Initialise the environment
    env = gym.make(f"MoonlanderWorld-collect-gaussian_with_distance-{difficulty}-benchmark-v0")
    
    reward_list, normalized_reward_list = run_through_env(env)
    
    plot_histogram_collect(reward_list=reward_list, task_name='Collect Asteroids')
    # plot_histogram_collect(reward_list=normalized_reward_list, task_name='Collect Asteroids Normalized')
