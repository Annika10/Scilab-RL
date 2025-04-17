import numpy as np
import torch
import math
import copy

from src.custom_envs.moonlander.utils import get_observation_of_position_and_object_positions, \
    get_next_position_observation_moonlander, get_collected_objects
from src.custom_envs.moonlander.helper_functions import calculate_gaussian_reward, \
    calculate_gaussian_with_distance_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_prediction_error(next_obs_positions, predicted_next_obs_positions, first_possible_x_position: int,
                               last_possible_x_position: int, linear_or_tanh: str = "linear") -> float:
    """
    Calculate the prediction error between the next obs and the predicted next obs
    Args:
        next_obs_positions: observation in positions after actually executing the action
        predicted_next_obs_positions: prediction of next observation
        first_possible_x_position: first possible x position of the agent
        last_possible_x_position: last possible x position of the agent
        linear_or_tanh: if the prediction error should be calculated by linear or tanh
    Returns:
        prediction error (number between 0 and 1) between the next obs and the predicted obs
    """
    # we just care for the x position of the moonlander agent,
    # because the y position is always equally to the size of the agent
    
    # euclidean distance
    x_distance = math.sqrt(torch.sum((predicted_next_obs_positions[0][0] - next_obs_positions[0][0]) ** 2))
    
    if linear_or_tanh == "linear":
        # maximal distance possible in moonlander world to use min/max scaler
        max_distance_in_moonlander_world = math.sqrt((last_possible_x_position - first_possible_x_position) ** 2)
        prediction_error = x_distance / max_distance_in_moonlander_world
    elif linear_or_tanh == "tanh":
        prediction_error = math.tanh(0.5 * x_distance)
    else:
        raise ValueError(f"The prediction error {linear_or_tanh} calculation is not supported. "
                         f"Prediction error can only be calculated by linear or tanh.")
    
    return prediction_error


def calculate_need_for_control(last_observation_positions: torch.Tensor, policy, observation_height: int,
                               observation_width: int, agent_size: int, task: str, object_dict_list: list[dict],
                               weighted: bool = True) -> tuple[float, float, float]:
    """
    Calculate the need for control of the environment by simulating the default trajectory
    and the "optimal" trajectory the agent would choose.
    Args:
        last_observation_positions: last observation positions
        policy: agent policy to predict actions
        observation_height: height of the observation
        observation_width: width of the observation
        agent_size: size of the agent in the observation
        task: task of the environment, either "dodge" or "collect"
        object_dict_list: list of objects in the environment
        weighted: if each predicted state is weighted fewer
    Returns:
        need for control between 0 and 1
        summed up rewards when executing the default action
        summed up rewards when executing the optimal action
    """
    
    if not task == "collect":
        raise ValueError(f"The current task {task} is not supported.")
    
    last_observation_positions_default = last_observation_positions
    last_observation_positions_optimal = copy.deepcopy(last_observation_positions_default)
    
    ##### CALCULATE REWARDS THROUGH ENVIRONMENT #####
    summed_up_reward_default = 0
    summed_up_reward_optimal = 0
    gamma = 0.9
    
    # simulate at least one step
    for i in range(max(observation_height, 1)):
        ##### DEFAULT ACTION #####
        # get next state
        last_observation_state_default, last_observation_positions_default = get_next_observation_as_state_and_positions(
            last_observation_positions=last_observation_positions_default,
            action=torch.tensor([1]),
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size,
            task=task)
        reward_default = get_next_reward(
            new_observation_state=last_observation_state_default,
            new_observation_positions=last_observation_positions_default,
            action=torch.tensor([1]),
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size,
            task=task,
            object_dict_list=object_dict_list)
        if not weighted:
            summed_up_reward_default += reward_default
        else:
            summed_up_reward_default += reward_default * math.pow(gamma, i)
        
        ##### OPTIMAL ACTION #####
        # get optimal action of agent
        action_of_task_agent, _ = policy.predict(last_observation_positions_optimal.cpu(), deterministic=True)
        # get next state
        last_observation_state_optimal, last_observation_positions_optimal = get_next_observation_as_state_and_positions(
            last_observation_positions=last_observation_positions_optimal,
            action=torch.tensor(action_of_task_agent),
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size,
            task=task)
        reward_optimal = get_next_reward(
            new_observation_state=last_observation_state_optimal,
            new_observation_positions=last_observation_positions_optimal,
            action=torch.tensor(action_of_task_agent),
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size,
            task=task,
            object_dict_list=object_dict_list)
        if not weighted:
            summed_up_reward_optimal += reward_optimal
        else:
            summed_up_reward_optimal += reward_optimal * math.pow(gamma, i)
    
    # distance between the two trajectories
    # this can happen because of small left or right movements
    if summed_up_reward_optimal < summed_up_reward_default:
        need_for_control = 0
    else:
        difference = summed_up_reward_optimal - summed_up_reward_default
        need_for_control = min(max(difference / 100, 0), 1)
    
    # need for control is high if the rewards are quite different
    return need_for_control, summed_up_reward_default, summed_up_reward_optimal


def normalize_gaussian_with_distance_reward(task: str, absolute_reward) -> float:
    # normalize reward with MinMaxScaler
    if task == "collect":
        one_percent_boundary = -200
        ninty_nine_percent_boundary = 400
        
        # 99% of the numbers are between -200 and 400
        if absolute_reward > ninty_nine_percent_boundary:
            absolute_reward = ninty_nine_percent_boundary
        elif absolute_reward < one_percent_boundary:
            absolute_reward = one_percent_boundary
        normalized_reward = (absolute_reward - one_percent_boundary) / (
                ninty_nine_percent_boundary - one_percent_boundary)
    else:
        raise NotImplementedError("Task {} not implemented".format(task))
    
    return normalized_reward


def calculate_trajectory_length(observation_height: int, prediction_error: float) -> float:
    # calculate the trajectory lengths through the prediction error
    # we decide that the trajectory length is half the observation size of the environment
    # when the prediction error is 0
    return - (observation_height / 2) * prediction_error + observation_height / 2


def get_next_observation_as_state_and_positions(last_observation_positions: torch.tensor, action: torch.Tensor,
                                                observation_width: int, observation_height: int, agent_size: int,
                                                task: str) -> tuple[torch.tensor, torch.tensor]:
    # remove already overlapping objects
    if task == "collect":
        collected_objects_of_last_state = get_collected_objects(observation_positions=last_observation_positions,
                                                                agent_size=agent_size,
                                                                observation_width=observation_width)
        for index in range(2, len(last_observation_positions[0]), 2):
            x_position = int(last_observation_positions[0][index])
            y_position = int(last_observation_positions[0][index + 1])
            for collected_object in collected_objects_of_last_state:
                if x_position == collected_object['x'] and y_position == collected_object['y']:
                    last_observation_positions[0][index] = 0
                    last_observation_positions[0][index + 1] = 0
    else:
        raise NotImplementedError("The task {} is not supported.".format(task))
    
    # get next positions for new step with action
    new_observation_positions = get_next_position_observation_moonlander(
        observations=last_observation_positions,
        actions=action,
        observation_width=observation_width,
        agent_size=agent_size)
    
    # determine state of new positions for reward calculation
    new_observation_state = np.expand_dims(
        get_observation_of_position_and_object_positions(agent_and_object_positions=new_observation_positions,
                                                         observation_height=observation_height,
                                                         observation_width=observation_width,
                                                         agent_size=agent_size,
                                                         task=task).flatten().cpu().numpy(),
        axis=0)
    
    return new_observation_state, new_observation_positions


def get_next_reward(new_observation_state: torch.Tensor, new_observation_positions: torch.tensor,
                    action: torch.Tensor, observation_width: int, observation_height: int, agent_size: int,
                    task: str, object_dict_list: list[dict]) -> float:
    if task == "collect":
        task_type = "coin"
    else:
        raise ValueError(f"The current task {task} is not supported.")
    
    # get collected objects to calculate reward
    x_position_of_agent = int(
        min(max(agent_size, new_observation_positions[0][0]), observation_width - agent_size + 1))
    y_position_of_agent = int(new_observation_positions[0][1])
    collected_objects = get_collected_objects(observation_positions=new_observation_positions, agent_size=agent_size,
                                              observation_width=observation_width)
    
    # calculate reward
    rewards_gaussian, _ = calculate_gaussian_reward(
        state=new_observation_state.reshape(observation_height, observation_width + 2),
        collected_objects=collected_objects,
        agent_size=agent_size,
        task_type=task_type,
        x_position_of_agent=x_position_of_agent,
        y_position_of_agent=y_position_of_agent)
    reward_with_distance = calculate_gaussian_with_distance_reward(x_position_of_agent=x_position_of_agent,
                                                                   y_position_of_agent=y_position_of_agent,
                                                                   following_observation_size=observation_height,
                                                                   agent_size=agent_size, task=task,
                                                                   action=action.item(),
                                                                   reward_gaussian=rewards_gaussian,
                                                                   object_dict_list=object_dict_list)
    return reward_with_distance
