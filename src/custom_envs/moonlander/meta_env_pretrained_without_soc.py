import os
import csv
import ast
import yaml
import torch
import numpy as np
import gymnasium as gym
from gymnasium import logger as gymnasium_logger
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from matplotlib import pyplot as plt

from src.custom_envs import ROOT_DIR
from src.custom_algorithms.ppo_moonlander import PPO_MOONLANDER
from src.custom_envs.moonlander.positions_wrapper import PositionsWrapperEnv
from src.custom_envs.moonlander.utils import get_observation_of_position_and_object_positions, \
    get_next_position_observation_moonlander
from src.custom_algorithms.ppo_moonlander.utils import normalize_gaussian_with_distance_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaEnvPretrainedWithoutSoC(gym.Env):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(self,
                 collect_task_one_best_model_name: str, collect_task_two_best_model_name: str,
                 config_file_name_collect_task_one: str = None, config_file_name_collect_task_two: str = None,
                 list_of_object_dict_lists_collect_task_one_filename: str = None,
                 list_of_object_dict_lists_collect_task_two_filename: str = None,
                 render_mode=None,
                 input_noise_in_subtasks_one: bool = False, input_noise_in_subtasks_two: bool = False,
                 consecutive_frames: int = 1,
                 normalize_rewards: bool = False):
        
        # load configs of pretrained models
        if config_file_name_collect_task_one is None:
            config_path_collect_task_one = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "standard_config_second_task.yaml")
        else:
            config_path_collect_task_one = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        config_file_name_collect_task_one)
        if config_file_name_collect_task_two is None:
            config_path_collect_task_two = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "standard_config_second_task.yaml")
        else:
            config_path_collect_task_two = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        config_file_name_collect_task_two)
        
        with open(config_path_collect_task_one, "r") as file:
            config_collect_task_one = yaml.safe_load(file)
        with open(config_path_collect_task_two, "r") as file:
            config_collect_task_two = yaml.safe_load(file)
        
        if not config_collect_task_one == config_collect_task_two:
            gymnasium_logger.warn("Configurations are not the same")
        
        # load object dict lists
        list_of_object_dict_lists_collect_task_one = None
        list_of_object_dict_lists_collect_task_two = None
        
        if list_of_object_dict_lists_collect_task_one_filename:
            list_of_object_dict_lists_collect_task_one = []
            with open(ROOT_DIR / "moonlander" / list_of_object_dict_lists_collect_task_one_filename,
                      "r") as file:
                lines = csv.reader(file)
                for line in lines:
                    # first element is index
                    # second element is the object list
                    # form string to list of dictionaries
                    list_of_object_dict_lists_collect_task_one.append(ast.literal_eval(line[1]))
        
        if list_of_object_dict_lists_collect_task_two_filename:
            list_of_object_dict_lists_collect_task_two = []
            with open(ROOT_DIR / "moonlander" / list_of_object_dict_lists_collect_task_two_filename, "r") as file:
                lines = csv.reader(file)
                for line in lines:
                    # first element is index
                    # second element is the object list
                    # form string to list of dictionaries
                    list_of_object_dict_lists_collect_task_two.append(ast.literal_eval(line[1]))
        
        # define render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        ### ACTION SPACE ###
        # one action to decide which task to control
        # action 0 --> task 0 can be controlled
        # action 1 --> task 1 can be controlled
        self.action_space = gym.spaces.Discrete(2)
        
        ### LOAD THE PRETRAINED MODELS ###
        # define logger for loading pretrained models
        tmp_path = "/tmp/sb3_log/"
        self.logger = configure(tmp_path, ["stdout", "csv"])
        
        collect_task_one_task_difficulty = config_collect_task_one["world"]["difficulty"]
        collect_task_two_task_difficulty = config_collect_task_two["world"]["difficulty"]
        
        input_noise_str_one = "input-noise-" if input_noise_in_subtasks_one else ""
        input_noise_str_two = "input-noise-" if input_noise_in_subtasks_two else ""
        # environments for the pretrained models
        collect_task_one_env = make_vec_env(
            f"MoonlanderWorld-collect-gaussian_with_distance-{collect_task_one_task_difficulty}-{input_noise_str_one}v0",
            n_envs=1,
            wrapper_class=PositionsWrapperEnv,
            env_kwargs={"list_of_object_dict_lists": list_of_object_dict_lists_collect_task_one,
                        "config_file_name": config_path_collect_task_one})
        collect_task_two_env = make_vec_env(
            f"MoonlanderWorld-collect-gaussian_with_distance-{collect_task_two_task_difficulty}-{input_noise_str_two}v0",
            n_envs=1,
            wrapper_class=PositionsWrapperEnv,
            env_kwargs={"list_of_object_dict_lists": list_of_object_dict_lists_collect_task_two,
                        "config_file_name": config_path_collect_task_two})
        
        # Load the trained agents
        # FIXME: this is an ugly hack to load the trained agents
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               f"../../../policies/{collect_task_one_best_model_name}.zip"), "rb") as file:
            print("start loading agents", file)
            self.trained_collect_task_one = PPO_MOONLANDER.load(path=file, env=collect_task_one_env)
            self.trained_collect_task_one.set_logger(logger=self.logger)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               f"../../../policies/{collect_task_two_best_model_name}.zip"), "rb") as file:
            self.trained_collect_task_two = PPO_MOONLANDER.load(path=file, env=collect_task_two_env)
            self.trained_collect_task_two.set_logger(logger=self.logger)
            print("finish loading agents")
        
        # reset the envs
        # only one return value because DummyVecEnv only returns one observation
        self.state_of_collect_task_one = self.trained_collect_task_one.env.reset()
        self.state_of_collect_task_two = self.trained_collect_task_two.env.reset()
        
        self.object_dict_list_task_one = \
            self.trained_collect_task_one.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        self.object_dict_list_task_two = \
            self.trained_collect_task_two.env.env_method("get_wrapper_attr", "object_dict_list")[0]
        
        # because both tasks use the same configuration, we can use one of them
        self.observation_width = self.trained_collect_task_one.env.env_method("get_wrapper_attr", "observation_width")[
            0]
        self.observation_height = \
            self.trained_collect_task_one.env.env_method("get_wrapper_attr", "observation_height")[0]
        self.agent_size = self.trained_collect_task_one.env.env_method("get_wrapper_attr", "size")[0]
        self.maximum_number_of_objects = \
            collect_task_one_env.env_method("get_wrapper_attr", "maximum_number_of_objects")[0]
        
        if self.observation_width + 2 >= self.observation_height + self.agent_size:
            maximum_possible_value = self.observation_width + 2
        else:
            maximum_possible_value = self.observation_height + self.agent_size
        
        ### OBSERVATION SPACE ###
        self.observation_space = gym.spaces.Box(
            low=-self.agent_size,
            high=maximum_possible_value,
            shape=(2 * (self.maximum_number_of_objects * 2 + 2),),
            dtype=np.int64,
        )
        
        # set state
        self.state = np.concatenate((self.state_of_collect_task_one, self.state_of_collect_task_two), axis=0).flatten()
        
        self.current_task = 0
        if isinstance(consecutive_frames, int) and consecutive_frames > 0:
            self.consecutive_frames = consecutive_frames
        else:
            self.consecutive_frames = 1
        self.normalize_rewards = normalize_rewards
        
        # for rendering
        # FIXME: However, Agg does not open any display window.
        # Agg, is a non-interactive backend that can only write to files.
        # It is used on Linux, if Matplotlib cannot connect to either an X display or a Wayland display.
        # matplotlib.use('agg')
        plt.ion()
        self.fig, self.ax = plt.subplots()
        eximg = np.zeros((self.observation_height, self.observation_width * 2 + 4))
        eximg[0] = -10
        eximg[1] = 5
        self.observation_for_rendering = eximg
        self.im = self.ax.imshow(eximg)
    
    def step(self, action: int):
        # do at least one step
        last_state, reward_over_consecutive_frames, done, truncated, info_over_consecutive_frames = self.actual_step(
            action)
        action_of_current_task_agent_list = [info_over_consecutive_frames["action_of_current_task_agent"]]
        
        for i in range(self.consecutive_frames - 1):
            # step the environment
            last_state, reward, done, truncated, info = self.actual_step(action)
            reward_over_consecutive_frames += reward
            info_over_consecutive_frames["collect_task_one_collected_objects"] += info[
                "collect_task_one_collected_objects"]
            info_over_consecutive_frames["collect_task_two_collected_objects"] += info[
                "collect_task_two_collected_objects"]
            info_over_consecutive_frames["collect_task_one_reward"] += info["collect_task_one_reward"]
            info_over_consecutive_frames["collect_task_two_reward"] += info["collect_task_two_reward"]
            action_of_current_task_agent_list.append(info["action_of_current_task_agent"])
            if done:
                break
        
        info_over_consecutive_frames["action_of_current_task_agent"] = action_of_current_task_agent_list
        info_over_consecutive_frames["position_state"] = last_state
        
        return last_state, reward_over_consecutive_frames, done, truncated, info_over_consecutive_frames
    
    def actual_step(self, action: int):
        """
        action: selects the task
                0: first collect task
                1: second collect task
        """
        match action:
            
            case 0:
                # collect task one
                active_model = self.trained_collect_task_one
                active_last_state = self.state_of_collect_task_one
                inactive_model = self.trained_collect_task_two
                inactive_last_state = self.state_of_collect_task_two
                self.current_task = 0
            case 1:
                # collect task two
                active_model = self.trained_collect_task_two
                active_last_state = self.state_of_collect_task_two
                inactive_model = self.trained_collect_task_one
                inactive_last_state = self.state_of_collect_task_one
                self.current_task = 1
            case _:
                raise ValueError(f"Invalid action {action}")
        
        ### START ACTIVE TASK ###
        # predict next action
        action_of_task_agent, _ = active_model.predict(active_last_state, deterministic=True)
        
        # perform action
        active_new_state, active_reward, active_is_done, active_info = active_model.env.step(action_of_task_agent)
        ### END ACTIVE TASK ###
        
        ### START INACTIVE TASK ###
        
        # perform default action 1 in inactive task
        # but meta agent does not see actual state and reward
        actual_inactive_observation, actual_inactive_reward, inactive_is_done, actual_inactive_info = inactive_model.env.step(
            torch.tensor([1], device=device))
        
        # calculate next belief state
        inactive_next_belief_state = get_next_position_observation_moonlander(
            observations=torch.from_numpy(inactive_last_state), actions=torch.tensor([1]),
            observation_width=self.observation_width, agent_size=self.agent_size).cpu().detach().numpy()
        ### END INACTIVE TASK ###
        
        match action:
            
            case 0:
                # collect task one
                self.state_of_collect_task_one = active_new_state
                self.state_of_collect_task_two = inactive_next_belief_state
                collect_task_one_reward = normalize_gaussian_with_distance_reward(task="collect",
                                                                                  absolute_reward=active_reward) if self.normalize_rewards else active_reward
                collect_task_two_reward = normalize_gaussian_with_distance_reward(task="collect",
                                                                                  absolute_reward=actual_inactive_reward) if self.normalize_rewards else actual_inactive_reward
                collect_task_one_info = active_info
                collect_task_two_info = actual_inactive_info
            case 1:
                # collect task two
                self.state_of_collect_task_one = inactive_next_belief_state
                self.state_of_collect_task_two = active_new_state
                collect_task_one_reward = normalize_gaussian_with_distance_reward(task="collect",
                                                                                  absolute_reward=actual_inactive_reward) if self.normalize_rewards else actual_inactive_reward
                collect_task_two_reward = normalize_gaussian_with_distance_reward(task="collect",
                                                                                  absolute_reward=active_reward) if self.normalize_rewards else active_reward
                collect_task_one_info = actual_inactive_info
                collect_task_two_info = active_info
            case _:
                raise ValueError(f"Invalid action {action}")
        
        self.state = np.concatenate((self.state_of_collect_task_one, self.state_of_collect_task_two), axis=0).flatten()
        active_reward = normalize_gaussian_with_distance_reward(task="collect",
                                                                absolute_reward=active_reward) if self.normalize_rewards else active_reward
        
        return self.state, active_reward.item(), (active_is_done or inactive_is_done).item(), False, {
            "collect_task_one_collected_objects": collect_task_one_info[0]["number_of_crashed_or_collected_objects"],
            "collect_task_two_collected_objects": collect_task_two_info[0]["number_of_crashed_or_collected_objects"],
            "collect_task_one_reward": collect_task_one_reward.item(),
            "collect_task_two_reward": collect_task_two_reward.item(),
            "action_of_current_task_agent": action_of_task_agent}
    
    def render(self):
        gymnasium_logger.warn("This is not the observation, the meta agent receives")
        # image is numpy array shape (2520,) reshaped to (30, 84)
        
        # form positions into simplified image
        simplified_image_collect_task_one = get_observation_of_position_and_object_positions(
            agent_and_object_positions=torch.from_numpy(self.state_of_collect_task_one),
            observation_height=self.observation_height,
            observation_width=self.observation_width, agent_size=self.agent_size, task="collect")
        simplified_image_collect_task_two = get_observation_of_position_and_object_positions(
            agent_and_object_positions=torch.from_numpy(self.state_of_collect_task_two),
            observation_height=self.observation_height,
            observation_width=self.observation_width, agent_size=self.agent_size, task="collect")
        
        merged_simplified_image = torch.cat(
            (simplified_image_collect_task_one.view(self.observation_height, self.observation_width + 2),
             simplified_image_collect_task_two.view(self.observation_height, self.observation_width + 2)),
            dim=1).flatten()
        
        merged_simplified_image = merged_simplified_image.reshape(
            (self.observation_height, self.observation_width * 2 + 4))
        
        # place frame around current task
        if self.current_task == 0:
            first_fill_value = -10
            second_fill_value = -1
            tmp = np.array([-10, -1])
            # +2 for walls + 2 for frame
            row = np.expand_dims(np.repeat(tmp, self.observation_width + 4), axis=0)
        else:
            first_fill_value = -1
            second_fill_value = -10
            tmp = np.array([-1, -10])
            # +2 for walls + 2 for frame
            row = np.expand_dims(np.repeat(tmp, self.observation_width + 4), axis=0)
        
        merged_simplified_image = np.concatenate(
            (
                np.full((self.observation_height, 1), first_fill_value),
                merged_simplified_image[:, :self.observation_width + 2],
                np.full((self.observation_height, 1), first_fill_value),
                np.full((self.observation_height, 1), second_fill_value),
                merged_simplified_image[:, self.observation_width + 2:],
                np.full((self.observation_height, 1), second_fill_value)),
            axis=1)
        merged_simplified_image = np.concatenate((row, merged_simplified_image, row), axis=0)
        self.observation_for_rendering = merged_simplified_image
        
        self.im.set_data(merged_simplified_image)
        if self.render_mode == "human":
            self.fig.canvas.draw_idle()
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # reset the envs
        # only one return value because DummyVecEnv only returns one observation
        # sub-agents are automatically reset in vectorized envs when doing the last step
        # remove reset there because otherwise, we don't have the 100 test episodes
        # self.state_of_collect_task_one = self.trained_collect_task_one.env.reset()
        # self.state_of_collect_task_two = self.trained_collect_task_two.env.reset()
        
        # set state
        self.state = np.concatenate((self.state_of_collect_task_one, self.state_of_collect_task_two), axis=0).flatten()
        
        self.current_task = 0
        
        return self.state, {}
