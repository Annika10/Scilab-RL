import gymnasium as gym
import numpy as np
import torch
from src.custom_envs.moonlander.meta_env_pretrained_without_soc import MetaEnvPretrainedWithoutSoC
from src.custom_algorithms.ppo_moonlander.utils import calculate_prediction_error, calculate_need_for_control
from src.custom_envs.moonlander.utils import get_next_position_observation_moonlander, get_collected_objects


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
            shape=(2,),
            dtype=np.float64,
        )
        
        self.object_dict_list_task_one = \
            self.env.trained_collect_task_one.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        self.object_dict_list_task_two = \
            self.env.trained_collect_task_two.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        
        # set state
        self.SoC_collect_task_one = 1.0
        self.SoC_collect_task_two = 1.0
        self.state = np.array([self.SoC_collect_task_one, self.SoC_collect_task_two])
    
    def step(self, action):
        """
        action: selects the task
                0: first collect task
                1: second collect task
        """
        ### REWARD ###
        # FIXME: reward is based on the last observation, not the new one
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
        
        match action:
            
            case 0:
                # collect task one
                active_model = self.env.trained_collect_task_one
                active_last_state = self.env.state_of_collect_task_one
                inactive_SoC = self.SoC_collect_task_two
                current_object_dict_list = self.object_dict_list_task_one
            case 1:
                # collect task two
                active_model = self.env.trained_collect_task_two
                active_last_state = self.env.state_of_collect_task_two
                inactive_SoC = self.SoC_collect_task_one
                current_object_dict_list = self.object_dict_list_task_two
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
        active_next_belief_state = torch.from_numpy(active_last_state)
        for action_of_current_task_agent in info["action_of_current_task_agent"]:
            # remove already overlapping objects
            collected_objects_of_last_state = get_collected_objects(observation_positions=active_next_belief_state,
                                                                    agent_size=self.env.agent_size,
                                                                    observation_width=self.env.observation_width)
            for index in range(2, len(active_next_belief_state[0]), 2):
                x_position = int(active_next_belief_state[0][index])
                y_position = int(active_next_belief_state[0][index + 1])
                for collected_object in collected_objects_of_last_state:
                    if x_position == collected_object['x'] and y_position == collected_object['y']:
                        active_next_belief_state[0][index] = 0
                        active_next_belief_state[0][index + 1] = 0
            active_next_belief_state = get_next_position_observation_moonlander(
                observations=active_next_belief_state,
                actions=torch.from_numpy(action_of_current_task_agent),
                observation_width=self.env.observation_width, agent_size=self.env.agent_size)
        
        prediction_error = calculate_prediction_error(next_obs_positions=np.expand_dims(active_new_state, axis=0),
                                                      predicted_next_obs_positions=active_next_belief_state,
                                                      first_possible_x_position=self.env.agent_size,
                                                      last_possible_x_position=self.env.observation_width - self.env.agent_size + 1,
                                                      linear_or_tanh="tanh")
        
        need_for_control, _, _ = calculate_need_for_control(
            last_observation_positions=torch.from_numpy(active_last_state),
            policy=active_model,
            observation_height=self.env.observation_height,
            observation_width=self.env.observation_width,
            agent_size=self.env.agent_size,
            task="collect",
            object_dict_list=current_object_dict_list,
            weighted=True)
        # prediction error is high, if the prediction and actual observation do not match
        # need for control is high if the rewards of the optimal trajectory are quite different to the rewards of the default trajectory
        # soc = mean of prediction error and need_for_control
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
        
        info["prediction_error"] = prediction_error
        info["need_for_control"] = need_for_control
        
        self.state = np.array([self.SoC_collect_task_one, self.SoC_collect_task_two])
        
        return self.state, reward, done, truncated, info
    
    def render(self):
        self.env.render()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        positions_state, info = self.env.reset()
        
        self.object_dict_list_task_one = \
            self.env.trained_collect_task_one.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        self.object_dict_list_task_two = \
            self.env.trained_collect_task_two.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        
        self.SoC_collect_task_one = 1.0
        self.SoC_collect_task_two = 1.0
        self.state = np.array([self.SoC_collect_task_one, self.SoC_collect_task_two])
        return self.state, info
