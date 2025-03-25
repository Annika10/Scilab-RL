import sys

import gymnasium as gym
import numpy as np
import torch
from src.custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv
from src.custom_algorithms.cleanppofm.utils import get_position_and_object_positions_of_observation, \
    get_next_position_observation_moonlander, get_observation_of_position_and_object_positions

torch.set_printoptions(threshold=sys.maxsize)


class ModelBasedWrapperEnv(gym.Env):
    """
    A class for wrapping the moonlander environment that turns concatenates the observation with predicted next observation
    This is needed because the observation space is used to define the input size of the policy network
    """
    
    def __init__(self, env):
        self.env = env
        if not isinstance(self.env.unwrapped, MoonlanderWorldEnv):
            raise NotImplementedError(
                f"This ModelBasedWrapperEnv is not implemented for the environment {self.env.unwrapped} yet!")
        
        self.action_space = env.action_space
        self.observation_space = gym.spaces.Box(
            low=-10,
            high=3,
            shape=(self.env.observation_height * (self.env.observation_width + 2) * 4,),
            dtype=np.int64,
        )
    
    def simulate_all_actions(self, state: np.array) -> np.array:
        # TODO: add this for bigger batch sizes
        
        # # get positions out of "image" observation
        # positions_observations = get_position_and_object_positions_of_observation(
        #     obs=rollout_data.observations, maximum_number_of_objects=10, observation_width=40,
        #     observation_height=30, agent_size=2)
        #
        # # duplicate positions 3 times for each action
        # positions_observations = positions_observations.repeat_interleave(3, dim=0)
        #
        # # actions
        # actions_0 = th.full((self.batch_size,), 0)
        # actions_1 = th.full((self.batch_size,), 1)
        # actions_2 = th.full((self.batch_size,), 2)
        # all_actions = th.stack((actions_0, actions_1, actions_2), dim=1).flatten()
        #
        # # get next position observation
        # next_positions_observations = get_next_position_observation_moonlander(
        #     observations=positions_observations, actions=all_actions, observation_width=40, agent_size=2)
        #
        # # get observation out of positions
        # next_observations = get_observation_of_position_and_object_positions(
        #     agent_and_object_positions=next_positions_observations, observation_width=40, observation_height=30,
        #     agent_size=2, task="collect")
        #
        # # link actual observation to next observations
        # concat_observation = th.cat(
        #     (rollout_data.observations.unsqueeze(1), next_observations.view(self.batchsize, 3, 1260)), dim=1).view(self.batchsize, -1)
        
        # get positions out of "image" observation
        positions_observations = get_position_and_object_positions_of_observation(
            obs=torch.tensor(state).unsqueeze(0), maximum_number_of_objects=10,
            observation_width=self.env.observation_width,
            observation_height=self.env.observation_height, agent_size=self.env.size)
        
        # duplicate positions 3 times for each action
        positions_observations = positions_observations.repeat_interleave(3, dim=0)
        
        # actions
        all_actions = torch.tensor([i for i in range(self.env.action_space.n)])
        
        # get next position observation
        next_positions_observations = get_next_position_observation_moonlander(
            observations=positions_observations, actions=all_actions, observation_width=self.env.observation_width,
            agent_size=self.env.size)
        
        # get observation out of positions
        next_observations = get_observation_of_position_and_object_positions(
            agent_and_object_positions=next_positions_observations, observation_width=self.env.observation_width,
            observation_height=self.env.observation_height, agent_size=self.env.size, task=self.env.task)
        
        # link actual observation to next observations
        return torch.cat((torch.tensor(state).unsqueeze(0), next_observations),
                         dim=0).flatten().cpu().detach().numpy().astype(np.int64)
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        return self.simulate_all_actions(state=state), reward, done, truncated, info
    
    def render(self, mode="human"):
        self.env.render(mode=mode)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, info = self.env.reset()
        return self.simulate_all_actions(state=state), info
