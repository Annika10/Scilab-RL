import gymnasium as gym
from src.custom_envs.moonlander.meta_env_pretrained_with_soc_wrapper import SoCObsAndRewardWrapperEnv


class SoCObsOnlyWrapperEnv(gym.Env):
    """
    A class for wrapping the meta environment pretrained already wrapped in the SoCRewardOnlyWrapperEnv and SoCObsAndRewardWrapperEnv
    and using the SoC in the observation and the active reward
    """
    
    def __init__(self, env):
        self.env = env
        if not isinstance(self.env.unwrapped, SoCObsAndRewardWrapperEnv):
            raise NotImplementedError(
                f"This SoCObsOnlyWrapperEnv is not implemented for the environment {self.env.unwrapped} yet!")
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.state = self.env.state
    
    def step(self, action):
        """
        action: selects the task
                0: first collect task
                1: second collect task
        """
        self.state, reward_soc, done, truncated, info = self.env.step(action)
        reward_gamescore = info["reward_gamescore"]
        
        return self.state, reward_gamescore, done, truncated, info
    
    def render(self):
        self.env.render()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state, info = self.env.reset()
        return self.state, info
