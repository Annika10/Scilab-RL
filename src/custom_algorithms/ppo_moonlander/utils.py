import numpy as np
import torch
import math
import copy

from src.custom_envs.moonlander.utils import get_position_and_object_positions_of_observation, \
    get_observation_of_position_and_object_positions, get_next_position_observation_moonlander, get_collected_objects
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


def calculate_need_for_control(last_observation_positions: torch.Tensor, policy, prediction_error: float,
                               observation_height: int, observation_width: int, agent_size: int, task: str,
                               object_dict_list: list[dict], maximum_number_of_objects: int = 10) -> tuple[
    float, float, float]:
    """
    Calculate the need for control of the environment by simulating the default trajectory
    and the "optimal" trajectory the agent would choose.
    Args:
        last_observation_positions: last observation positions
        policy: agent policy to predict actions
        prediction_error: error between predicted and last actual observation
        observation_height: height of the observation
        observation_width: width of the observation
        agent_size: size of the agent in the observation
        task: task of the environment, either "dodge" or "collect"
        object_dict_list: list of objects in the environment
        maximum_number_of_objects: the number of objects that are considered in the forward model prediction
    Returns:
        need for control between 0 and 1
        summed up rewards when executing the default action (trajectory length is calculated by prediction error)
    """
    
    if task == "dodge":
        task_type = "obstacle"
    elif task == "collect":
        task_type = "coin"
    else:
        raise ValueError(f"The current task {task} is not supported.")
    
    # calculate the trajectory lengths through the prediction error
    # we decide that the trajectory length is half the observation size of the environment
    # when the prediction error is 0
    trajectory_length = calculate_trajectory_length(observation_height=observation_height,
                                                    prediction_error=prediction_error)
    
    last_observation_positions_default = last_observation_positions
    last_observation_positions_optimal = copy.deepcopy(last_observation_positions_default)
    
    ##### CALCULATE REWARDS THROUGH ENVIRONMENT #####
    summed_up_reward_default = 0
    summed_up_reward_optimal = 0
    summed_up_reward_default_weighted = 0
    summed_up_reward_optimal_weighted = 0
    gamma = 0.9
    
    # simulate at least one step
    for i in range(max(round(trajectory_length), 1)):
        ##### DEFAULT ACTION #####
        # get next state
        last_observation_state_default, last_observation_positions_default = get_next_observation_as_state_and_positions(
            last_observation_positions=last_observation_positions_default,
            action=torch.tensor([1]),
            maximum_number_of_objects=maximum_number_of_objects,
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size,
            task=task)
        normalized_reward_default = get_next_normalized_reward(
            last_observation_state=last_observation_state_default,
            last_observation_positions=last_observation_positions_default,
            action=torch.tensor([1]),
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size,
            task=task,
            object_dict_list=object_dict_list)
        summed_up_reward_default += normalized_reward_default
        summed_up_reward_default_weighted += normalized_reward_default * math.pow(gamma, i)
        
        ##### OPTIMAL ACTION #####
        # get optimal action of agent
        action_of_task_agent, _ = policy.predict(last_observation_positions_optimal, deterministic=True)
        # get next state
        last_observation_state_optimal, last_observation_positions_optimal = get_next_observation_as_state_and_positions(
            last_observation_positions=last_observation_positions_optimal,
            action=torch.tensor(action_of_task_agent),
            maximum_number_of_objects=maximum_number_of_objects,
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size,
            task=task)
        normalized_reward_optimal = get_next_normalized_reward(
            last_observation_state=last_observation_state_optimal,
            last_observation_positions=last_observation_positions_optimal,
            action=torch.tensor(action_of_task_agent),
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size,
            task=task,
            object_dict_list=object_dict_list)
        summed_up_reward_optimal += normalized_reward_optimal
        summed_up_reward_optimal_weighted += normalized_reward_optimal * math.pow(gamma, i)
    
    # get a mean reward between 0 and 1
    summed_up_reward_default_normalized = summed_up_reward_default / (max(round(trajectory_length), 1))
    summed_up_reward_optimal_normalized = summed_up_reward_optimal / (max(round(trajectory_length), 1))
    summed_up_reward_default_weighted_normalized = summed_up_reward_default_weighted / (
        max(round(trajectory_length), 1))
    summed_up_reward_optimal_weighted_normalized = summed_up_reward_optimal_weighted / (
        max(round(trajectory_length), 1))
    
    # distance between the two trajectories
    need_for_control = (max(summed_up_reward_default_normalized, summed_up_reward_optimal_normalized)) - (
        min(summed_up_reward_default_normalized, summed_up_reward_optimal_normalized))
    need_for_control_weighted = (max(summed_up_reward_default_weighted_normalized,
                                     summed_up_reward_optimal_weighted_normalized)) - (
                                    min(summed_up_reward_default_weighted_normalized,
                                        summed_up_reward_optimal_weighted_normalized))
    
    # need for control is high if the rewards are quite different
    return need_for_control_weighted, summed_up_reward_default_weighted_normalized, summed_up_reward_optimal_weighted_normalized


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


def get_next_observation_as_state_and_positions(action: torch.Tensor, maximum_number_of_objects: int,
                                                observation_width: int, observation_height: int, agent_size: int,
                                                task: str, last_observation_state: torch.Tensor = None,
                                                last_observation_positions: torch.tensor = None) -> tuple[
    torch.tensor, torch.tensor]:
    if last_observation_state is not None:
        if not last_observation_state.shape[1] == (observation_width + 2) * observation_height:
            raise ValueError(
                f"The given observation width {observation_width} and height {observation_height} "
                f"do not match the observation shape: {last_observation_state.shape}."
                f"The second observation shape element {last_observation_state.shape[1]} should be "
                f"(observation_width + 2) * observation_height = {(observation_width + 2) * observation_height}.")
        
        # get positions of last observation
        last_observation_positions = get_position_and_object_positions_of_observation(
            torch.tensor(last_observation_state, device=device), maximum_number_of_objects=maximum_number_of_objects,
            observation_width=observation_width, observation_height=observation_height, agent_size=agent_size)
    elif last_observation_state is None and last_observation_positions is None:
        raise ValueError("Either last_observation_state or last_observation_positions should be given.")
    
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


def get_next_normalized_reward(last_observation_state: torch.Tensor, last_observation_positions: torch.tensor,
                               action: torch.Tensor, observation_width: int, observation_height: int, agent_size: int,
                               task: str,
                               object_dict_list: list[dict]) -> float:
    if task == "collect":
        task_type = "coin"
    else:
        raise ValueError(f"The current task {task} is not supported.")
    
    # get collected objects to calculate reward
    x_position_of_agent = int(
        min(max(agent_size, last_observation_positions[0][0]), observation_width - agent_size + 1))
    y_position_of_agent = int(last_observation_positions[0][1])
    collected_objects = get_collected_objects(observation_positions=last_observation_positions, agent_size=agent_size,
                                              observation_width=observation_width)
    
    # calculate reward
    rewards_gaussian, _ = calculate_gaussian_reward(
        state=last_observation_state.reshape(observation_height, observation_width + 2),
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
    
    # FIXME: normalize reward???
    normalized_reward = normalize_gaussian_with_distance_reward(task=task, absolute_reward=reward_with_distance)
    
    return normalized_reward
