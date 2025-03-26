import gymnasium as gym
import numpy as np
from src.custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv


class ImageWrapperEnv(gym.Env):
    """
    A class for wrapping the moonlander environment that turns the output into an image.
    """
    
    def __init__(self, env):
        self.env = env
        if not isinstance(self.env.unwrapped, MoonlanderWorldEnv):
            raise NotImplementedError(
                f"This ImageWrapper is not implemented for the environment {self.env.unwrapped} yet!")
        
        self.action_space = env.action_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            # Same shape as the moonlander environment, but with a channel for a grayscale image added
            # This was removed: multiplied by ten because otherwise the image gets to small in the CNN
            shape=(1, env.following_observation_size, (env.config["world"]["x_width"] + 2)),
            dtype=np.uint8,
        )
    
    def to_image(self, state: np.array) -> np.array:
        """
        Args:
            state: A state from the moonlander environment

        Returns: The same state with the values of the matrix rescaled to [0, 255] and duplicated across three
                channels, which can be used as an image.
        """
        
        state = state.reshape((self.env.following_observation_size, self.env.config["world"]["x_width"] + 2))
        image = np.full((self.env.following_observation_size, (self.env.config["world"]["x_width"] + 2), 1),
                        255, dtype=np.uint8)
        # make agent black
        image[state == 1] = 0
        # make coins green
        image[state == 2] = 64
        # make obstacles red
        image[state == 3] = 128
        # make walls blue
        image[state == -1] = 191
        # make crashes yellow
        image[state == -10] = 255
        
        # channel-first is used for stable baselines
        # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        image_channel_first = np.transpose(image, (2, 0, 1))
        
        return image_channel_first
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        return self.to_image(state=state), reward, done, truncated, info
    
    def render(self, mode="human"):
        raise NotImplementedError("Rendering is not implemented for image observations in MoonlanderWorldEnv")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, info = self.env.reset()
        return self.to_image(state=state), info
