import gymnasium as gym
import numpy as np
from src.custom_envs.moonlander.meta_env_pretrained_with_soc_reward_only_wrapper import SoCRewardOnlyWrapperEnv


class SoCObsAndRewardWrapperEnv(gym.Env):
    """
    A class for wrapping the meta environment pretrained already wrapped in the SoCRewardOnlyWrapperEnv
    and using the SoC in the observation AND reward
    """
    
    def __init__(self, env):
        self.env = env
        if not isinstance(self.env.unwrapped, SoCRewardOnlyWrapperEnv):
            raise NotImplementedError(
                f"This SoCObsAndRewardWrapperEnv is not implemented for the environment {self.env.unwrapped} yet!")
        
        # NfC task one, reward task one, NfC task two, reward task two (two are predicted)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(2,),
            dtype=np.float64,
        )
        
        self.state = np.array([self.env.SoC_collect_task_one, self.env.SoC_collect_task_two])
    
    def step(self, action):
        """
        action: selects the task
                0: first collect task
                1: second collect task
        """
        positions_state, reward_soc, done, truncated, info = self.env.step(action)
        
        self.state = np.array([self.env.SoC_collect_task_one, self.env.SoC_collect_task_two])
        
        return self.state, reward_soc, done, truncated, info
    
    def render(self):
        self.env.render()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        positions_state, info = self.env.reset()
        
        self.state = np.array([self.env.SoC_collect_task_one, self.env.SoC_collect_task_two])
        return self.state, info
