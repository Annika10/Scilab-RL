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
            # Same shape as the moonlander environment, but with RGB channels added
            # and multiplied by ten because otherwise the image gets to small in the CNN
            shape=(
                3,
                env.following_observation_size * 10,
                (env.config["world"]["x_width"] + 2) * 10
            ),
            dtype=np.uint8,
        )
    
    def to_image(self, state: np.array) -> np.array:
        """
        Args:
            state: A state from the moonlander environment

        Returns: The same state with the values of the matrix rescaled to [0, 255] and duplicated across three
                channels, which can be used as an image.
        """
        
        # we have to upscale our state because the CNN in the policy has to big kernel size and reduces the image too much
        # Duplicate each entry by ten in both dimensions
        state = np.repeat(
            np.repeat(state.reshape((self.env.following_observation_size, self.env.config["world"]["x_width"] + 2)), 10,
                      axis=0), 10, axis=1)
        
        image = np.full((self.env.following_observation_size * 10, (self.env.config["world"]["x_width"] + 2) * 10, 3),
                        255, dtype=np.uint8)
        # make agent black
        image[state == 1] = 0
        # make coins green
        image[state == 2] = [0, 128, 0]
        # make obstacles red
        image[state == 3] = [255, 0, 0]
        # make walls blue
        image[state == -1] = [0, 0, 255]
        # make crashes yellow
        image[state == -10] = [255, 255, 0]
        
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
