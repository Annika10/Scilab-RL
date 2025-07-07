import gymnasium as gym
from src.custom_envs.moonlander.meta_env_pretrained_without_soc import MetaEnvPretrainedWithoutSoC
from src.custom_algorithms.ppo_moonlander.utils import normalize_gaussian_with_distance_reward


class SwitchingBoostWrapperEnv(gym.Env):
    """
    A class for wrapping the meta environment pretrained and using the SoC as a reward
    """
    
    def __init__(self, env, boost_value):
        self.env = env
        if not isinstance(self.env.unwrapped, MetaEnvPretrainedWithoutSoC):
            raise NotImplementedError(
                f"This SwitchingBoostWrapperEnv is not implemented for the environment {self.env.unwrapped} yet!")
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.boost_value = boost_value
        self.last_action = 0
    
    def step(self, action):
        """
        action: selects the task
                0: first collect task
                1: second collect task
        """
        self.state, reward, done, truncated, info = self.env.step(action)
        if not self.env.normalize_rewards:
            # get a reward between 0 and 1
            reward = normalize_gaussian_with_distance_reward(task="collect",
                                                             absolute_reward=reward / self.env.consecutive_frames)
        
        info["reward_without_boost"] = reward
        
        if self.last_action != action:
            reward += self.boost_value
        
        self.last_action = action
        return self.state, reward, done, truncated, info
    
    def render(self):
        self.env.render()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state, info = self.env.reset()
        
        return self.state, info
