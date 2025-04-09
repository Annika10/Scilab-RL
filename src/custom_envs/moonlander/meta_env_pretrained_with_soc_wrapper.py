import gymnasium as gym
import numpy as np
import torch
from src.custom_envs.moonlander.meta_env_pretrained_without_soc import MetaEnvPretrainedWithoutSoC
from src.custom_algorithms.ppo_moonlander.utils import calculate_prediction_error, calculate_need_for_control, \
    normalize_gaussian_with_distance_reward
from src.custom_envs.moonlander.utils import get_next_position_observation_moonlander


class SoCWrapperEnv(gym.Env):
    """
    A class for wrapping the meta environment pretrained that calculates the SoC and uses it as the observation
    """
    
    def __init__(self, env):
        self.env = env
        if not isinstance(self.env.unwrapped, MetaEnvPretrainedWithoutSoC):
            raise NotImplementedError(
                f"This SoCWrapperEnv is not implemented for the environment {self.env.unwrapped} yet!")
        
        self.action_space = env.action_space
        # NfC task one, reward task one, NfC task two, reward task two (two are predicted)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=np.float64,
        )
        
        self.object_dict_list_task_one = \
            self.env.trained_collect_task_one.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        self.object_dict_list_task_two = \
            self.env.trained_collect_task_two.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        
        # set state
        self.SoC_collect_task_one = 1.0
        self.SoC_collect_task_two = 1.0
        self.reward_collect_task_one = 0.0
        self.reward_collect_task_two = 0.0
        self.state = np.array([self.SoC_collect_task_one, self.reward_collect_task_one, self.SoC_collect_task_two,
                               self.reward_collect_task_two])
    
    def step(self, action):
        """
        action: selects the task
                0: first collect task
                1: second collect task
        """
        
        match action:
            
            case 0:
                # collect task one
                active_model = self.env.trained_collect_task_one
                active_last_state = self.env.state_of_collect_task_one
                inactive_SoC = self.SoC_collect_task_two
                current_object_dict_list = self.object_dict_list_task_one
                self.current_task = 0
            case 1:
                # collect task two
                active_model = self.env.trained_collect_task_two
                active_last_state = self.env.state_of_collect_task_two
                inactive_SoC = self.SoC_collect_task_one
                current_object_dict_list = self.object_dict_list_task_two
                self.current_task = 1
            case _:
                raise ValueError(f"Invalid action {action}")
        
        positions_state, reward_gamescore, done, truncated, info = self.env.step(action)
        
        match action:
            
            case 0:
                active_new_state = positions_state[:self.env.maximum_number_of_objects * 2 + 2]
            case 1:
                active_new_state = positions_state[self.env.maximum_number_of_objects * 2 + 2:]
            case _:
                raise ValueError(f"Invalid action {action}")
        
        # calculate next belief state
        active_next_belief_state = get_next_position_observation_moonlander(
            observations=torch.from_numpy(active_last_state),
            actions=torch.from_numpy(info["action_of_current_task_agent"]),
            observation_width=self.env.observation_width, agent_size=self.env.agent_size)
        
        # FIXME: change prediction error to be higher if the agent x position is wrong? --> tanh
        prediction_error = calculate_prediction_error(next_obs_positions=np.expand_dims(active_new_state, axis=0),
                                                      predicted_next_obs_positions=active_next_belief_state,
                                                      first_possible_x_position=self.env.agent_size,
                                                      last_possible_x_position=self.env.observation_width - self.env.agent_size + 1,
                                                      linear_or_tanh="linear")
        
        # what is about reward normalization after changing the reward function?
        need_for_control, _, _ = calculate_need_for_control(
            last_observation_positions=torch.from_numpy(active_last_state),
            policy=active_model,
            prediction_error=prediction_error,
            observation_height=self.env.observation_height,
            observation_width=self.env.observation_width,
            agent_size=self.env.agent_size,
            task="collect",
            object_dict_list=current_object_dict_list,
            maximum_number_of_objects=self.env.maximum_number_of_objects)
        # prediction error is high, if the prediction and actual observation do not match
        # need for control is high if the rewards of the optimal trajectory are quite different to the rewards of the default trajectory
        # soc = mean of prediction error and need_for_control
        # FIXME: SoC only NfC because PE is incorporated in the NfC?
        active_SoC = 1 - ((prediction_error + need_for_control) / 2)
        
        # SoC update --> degrade SoC by factor of observation height, so that after half of the steps of the observation
        # the SoC is 0.5 and after all steps the SoC is 0
        inactive_SoC = min(max(0.0, inactive_SoC - (1 / self.env.observation_height)), 1.0)
        
        match action:
            
            case 0:
                # collect task one
                self.SoC_collect_task_one = active_SoC
                self.SoC_collect_task_two = inactive_SoC
            case 1:
                # collect task two
                self.SoC_collect_task_one = inactive_SoC
                self.SoC_collect_task_two = active_SoC
            case _:
                raise ValueError(f"Invalid action {action}")
        
        self.reward_collect_task_one = normalize_gaussian_with_distance_reward(task="collect", absolute_reward=info[
            "collect_task_one_reward"])
        self.reward_collect_task_two = normalize_gaussian_with_distance_reward(task="collect", absolute_reward=info[
            "collect_task_two_reward"])
        
        self.state = np.array([self.SoC_collect_task_one, self.reward_collect_task_one, self.SoC_collect_task_two,
                               self.reward_collect_task_two])
        
        ### REWARD ###
        reward = 0
        if self.SoC_collect_task_one > self.SoC_collect_task_two:
            if action == 0:
                reward = -1
            else:
                reward = 1
        elif self.SoC_collect_task_one < self.SoC_collect_task_two:
            if action == 1:
                reward = -1
            else:
                reward = 1
        
        return self.state, reward, done, truncated, info
    
    def render(self, mode="human"):
        self.env.render(mode=mode)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        positions_state, info = self.env.reset()
        
        self.object_dict_list_task_one = \
            self.env.trained_collect_task_one.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        self.object_dict_list_task_two = \
            self.env.trained_collect_task_two.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        
        self.SoC_collect_task_one = 1.0
        self.SoC_collect_task_two = 1.0
        self.reward_collect_task_one = 0.0
        self.reward_collect_task_two = 0.0
        self.state = np.array([self.SoC_collect_task_one, self.reward_collect_task_one, self.SoC_collect_task_two,
                               self.reward_collect_task_two])
        return self.state, info
