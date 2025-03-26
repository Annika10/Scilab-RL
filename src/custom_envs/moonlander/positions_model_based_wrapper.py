import gymnasium as gym
import numpy as np
import torch
from src.custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv
from src.custom_algorithms.cleanppofm.utils import get_position_and_object_positions_of_observation, \
    get_next_position_observation_moonlander

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionsModelBasedWrapperEnv(gym.Env):
    """
    A class for wrapping the moonlander environment that turns the observation into positions instead of an image
    """
    
    def __init__(self, env, maximum_number_of_objects=10):
        self.env = env
        self.maximum_number_of_objects = maximum_number_of_objects
        if not isinstance(self.env.unwrapped, MoonlanderWorldEnv):
            raise NotImplementedError(
                f"This PositionsModelBasedWrapperEnv is not implemented for the environment {self.env.unwrapped} yet!")
        
        if self.env.observation_width + 2 >= self.env.observation_height + self.env.size:
            maximum_possible_value = self.env.observation_width + 2
        else:
            maximum_possible_value = self.env.observation_height + self.env.size
        
        self.action_space = env.action_space
        self.observation_space = gym.spaces.Box(
            low=-self.env.size,
            high=maximum_possible_value,
            shape=(1 + self.env.action_space.n, self.maximum_number_of_objects * 2 + 2),
            dtype=np.int64,
        )
    
    def simulate_all_actions(self, state: np.array) -> torch.Tensor:
        # squeeze to remove a dimension because we have no batches but one step
        position_state = get_position_and_object_positions_of_observation(
            obs=torch.tensor(state).unsqueeze(0), maximum_number_of_objects=self.maximum_number_of_objects,
            observation_width=self.env.observation_width, observation_height=self.env.observation_height,
            agent_size=self.env.size)
        
        # duplicate positions 3 times for each action
        position_state_for_three_actions = position_state.repeat_interleave(self.env.action_space.n, dim=0)
        
        # actions
        all_actions = torch.tensor([action for action in range(self.env.action_space.n)]).to(device=device)
        
        # get next position observation
        next_positions_observations = get_next_position_observation_moonlander(
            observations=position_state_for_three_actions, actions=all_actions,
            observation_width=self.env.observation_width, agent_size=self.env.size)
        
        # concat actual observation with next observations
        # numpy because gymnasium.Box does not accept torch tensors
        return torch.cat((position_state, next_positions_observations)).cpu().detach().numpy().astype(np.int64)
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        return self.simulate_all_actions(state=state), reward, done, truncated, info
    
    def render(self, mode="human"):
        self.env.render(mode=mode)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, info = self.env.reset()
        return self.simulate_all_actions(state=state), info
