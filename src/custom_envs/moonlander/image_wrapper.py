import gymnasium as gym
import numpy as np
from gym import Env


class ImageWrapperEnv(Env):
    """
    A class for wrapping the moonlander environment that turns the output into an image.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            # Same shape as the moonlander environment, but with RGB channels added
            shape=(
                env.following_observations_size,
                env.config["world"]["x_width"] + 2,
                3,
            ),
            dtype=np.uint8,
        )

    def to_image(self, state):
        """
        Args:
            state: A state from the moonlander environment

        Returns: The same state with the values of the matrix rescaled to [0, 255] and duplicated across three
                channels, which can be used as an image.
        """

        # Unflatten the state back into a 2D input
        state = np.resize(
            state,
            (
                self.env.following_observations_size,
                self.env.config["world"]["x_width"] + 2,
            ),
        )

        observation_space = self.env.observation_space

        min_value = observation_space.low[0]
        max_value = observation_space.high[0]

        value_range_length = abs(min_value) + abs(max_value)
        max_target_value = 255

        # Rescale values
        image_state = (
            max_target_value * (state - min_value) / value_range_length
        ).astype(int)

        # Add channel dimension (0 for NGE, 2 for dreamer)
        image_state = np.expand_dims(image_state, axis=2)

        # Turn into RGB instead of intensity image by duplicating along the channel axis
        image_state = np.repeat(image_state, 3, axis=2)

        return image_state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.to_image(state), reward, done, info

    def reset(self):
        return self.to_image(self.env.reset())

    def render(self, mode="human"):
        self.env.render(mode)
