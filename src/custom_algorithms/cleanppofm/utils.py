import numpy as np
import torch
import math
import copy

from src.custom_envs.moonlander.utils import get_position_and_object_positions_of_observation, \
    get_observation_of_position_and_object_positions, get_next_position_observation_moonlander, get_collected_objects
from src.custom_envs.moonlander.helper_functions import calculate_gaussian_reward, \
    calculate_gaussian_with_distance_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_obs(obs: dict) -> torch.Tensor:
    """
    Flatten a dict observation of the Gridworld envs.
    Args:
        obs: observation of type dict

    Returns:
        flattened observation in tensor format

    """
    # tensor can not check for string ("agent" in obs)
    if isinstance(obs, dict):
        # this is the flatten process for the Gridworld envs
        agent, target = obs['agent'], obs['target']
        if isinstance(agent, np.ndarray):
            agent = torch.from_numpy(agent).to(device)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).to(device)
        return torch.cat([agent, target], dim=1).to(dtype=torch.float32).detach().clone()
    else:
        raise NotImplementedError(
            "Flatten observation not implemented for this environment with this observation type.")


def layer_init(layer, std: np.float64 = np.sqrt(2), bias_const: float = 0.0) -> torch.nn.Module:
    """
    Initialize the layer with a (semi) orthogonal matrix and a constant bias.
    Args:
        layer: layer to be initialized
        std: standard deviation for the orthogonal matrix
        bias_const: constant for the bias

    Returns:
        initialized layer

    """
    # Fill the layer weight with a (semi) orthogonal matrix.
    # Described in Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    # Saxe, A. et al. (2013).
    torch.nn.init.orthogonal_(layer.weight, std)
    # Fill the layer bias with the value bias_const.
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_summed_up_reward_of_env_with_predicted_states_hardcoded(env, last_observation_positions: torch.Tensor,
                                                                number_of_future_steps: int = 10) -> float:
    """
    Get the reward of the forward model prediction or environment through predicted states of the forward model
     for number_of_future_steps steps.
    Args:
        env: environment
        last_observation_positions: last known observation in form of positions
        number_of_future_steps: number of future steps to predict

    Returns:
        summed up reward of the forward model or environment for number_of_future_steps steps

    """
    env_name = env.env_method("get_wrapper_attr", "name")[0]
    task = env.env_method("get_wrapper_attr", "task")[0]
    observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
    observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
    agent_size = env.env_method("get_wrapper_attr", "size")[0]
    
    # default action is stay at same position
    if env_name == "MoonlanderWorldEnv":
        default_action = torch.tensor([[1]]).to(device)
    # using reward model of other envs is not implemented by now
    else:
        raise NotImplementedError(
            f"The current environment has not implemented the manual reward calculation for {number_of_future_steps} steps.")
    
    if task == "dodge":
        task_type = "obstacle"
    elif task == "collect":
        task_type = "coin"
    else:
        raise ValueError(f"The current task {task} is not supported.")
    
    summed_up_reward = 0
    for i in range(number_of_future_steps):
        # get observation state of positions
        last_observation_state = np.expand_dims(
            get_observation_of_position_and_object_positions(agent_and_object_positions=last_observation_positions,
                                                             observation_height=observation_height,
                                                             observation_width=observation_width,
                                                             agent_size=agent_size,
                                                             task=task).flatten().cpu().numpy(),
            axis=0)
        
        x_position_of_agent = int(
            min(max(agent_size, last_observation_positions[0][0]), observation_width - agent_size + 1))
        y_position_of_agent = int(last_observation_positions[0][1])
        collected_objects = get_collected_objects(observation_positions=last_observation_positions,
                                                  agent_size=agent_size, observation_width=observation_width)
        
        # calculate reward
        rewards, _ = calculate_gaussian_reward(
            state=last_observation_state.reshape(observation_height, observation_width + 2),
            collected_objects=collected_objects,
            agent_size=agent_size,
            task_type=task_type,
            x_position_of_agent=x_position_of_agent,
            y_position_of_agent=y_position_of_agent)
        
        # normalize reward
        normalized_reward = normalize_rewards(task=task, absolute_reward=rewards)
        summed_up_reward += normalized_reward
        
        # problem: when collecting an object, the coin is removed after one step in the environment
        # here the coin stays the whole "collecting process" because we don't use the environment
        # therefore we get the collected objects and remove them, when they were collected
        if task == "collect":
            for collected_object in collected_objects:
                for index in range(2, len(last_observation_positions[0]), 2):
                    if collected_object['x'] == last_observation_positions[0][index] and collected_object['y'] == \
                            last_observation_positions[0][index + 1]:
                        last_observation_positions[0][index] = 0
                        last_observation_positions[0][index + 1] = 0
        
        # get next positions
        last_observation_positions = get_next_position_observation_moonlander(
            observations=last_observation_positions,
            actions=default_action[0],
            observation_width=observation_width,
            agent_size=agent_size)
    
    # normalize with mean of summed_up_reward
    return summed_up_reward / number_of_future_steps


def calculate_prediction_error_with_forward_model(env_name, next_obs_positions,
                                                  forward_model_prediction_normal_distribution: torch.normal,
                                                  first_possible_x_position: int,
                                                  last_possible_x_position: int) -> float:
    """
    Calculate the prediction error between the next obs and the forward model prediction.
    Args:
        env_name: name of the environment
        next_obs_positions: observation in positions after actually executing the action
        forward_model_prediction_normal_distribution: prediction of next observation by forward model
        first_possible_x_position: first possible x position of the agent
        last_possible_x_position: last possible x position of the agent

    Returns:
        prediction error between the next obs and the forward model prediction
    """
    ##### CALCULATE PREDICTION ERROR #####
    # prediction error version one -> standard deviation
    # prediction_error = forward_normal.stddev.mean().item()
    # prediction error version two -> Euclidean distance
    # calculate manually prediction error (Euclidean distance)
    if env_name == "MoonlanderWorldEnv":
        # we just care for the x position of the moonlander agent, because the y position is always equally
        # to the size of the agent
        # independently of using the whole obs or the position prediction,
        # we use the position predictions to calculate the prediction error
        
        # maximal distance possible in moonlander world to use min/max scaler
        max_distance_in_moonlander_world = math.sqrt(
            (last_possible_x_position - first_possible_x_position) ** 2)
        
        predicted_x_position = torch.tensor([min(max(first_possible_x_position,
                                                     forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[
                                                         0][0]),
                                                 last_possible_x_position)], device=device)
        prediction_error = (math.sqrt(
            torch.sum((predicted_x_position - next_obs_positions[0][0]) ** 2))) / max_distance_in_moonlander_world
    else:
        raise ValueError("Environment not supported")
    
    return prediction_error


def calculate_need_for_control_with_forward_model(env, policy, fm_network, logger, position_predicting: bool,
                                                  prediction_error: float,
                                                  maximum_number_of_objects: int = 5,
                                                  last_observation_state: np.array = None,
                                                  weighting: bool = False) -> tuple[
    float, float, float]:
    """
    Calculate the need for control of the environment by simulating the default trajectory
    and the "optimal" trajectory the agent would choose.
    Args:
        env: environment
        policy: agent policy to predict actions
        fm_network: forward model network
        logger: logger
        prediction_error: error between predicted and last actual observation
        position_predicting: if the forward model is predicting the position or actual observation
        maximum_number_of_objects: the number of objects that are considered in the forward model prediction
        last_observation_state: last observation state can be given and not selected by the environment
    Returns:
        need for control between 0 and 1
        summed up rewards when executing the default action (trajectory length is calculated by prediction error)
    """
    env_name = env.env_method("get_wrapper_attr", "name")[0]
    # default action is stay at same position
    if env_name == "MoonlanderWorldEnv":
        default_action = torch.tensor([[1]]).to(device)
    # using reward model of other envs is not implemented by now
    else:
        raise ValueError(
            "The current environment does not support the need for control calculation.")
    
    if not position_predicting:
        raise NotImplementedError("Using the actual states instead of positions is not implemented yet.")
    
    observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
    observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
    agent_size = env.env_method("get_wrapper_attr", "size")[0]
    task = env.env_method("get_wrapper_attr", "task")[0]
    
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
    
    if last_observation_state is None:
        last_observation_state_default = np.expand_dims(env.env_method("get_wrapper_attr", "state")[0].flatten(),
                                                        axis=0)
        last_observation_state_optimal = copy.deepcopy(last_observation_state_default)
    else:
        last_observation_state_default = last_observation_state
        last_observation_state_optimal = copy.deepcopy(last_observation_state_default)
    
    ##### CALCULATE REWARDS THROUGH ENVIRONMENT #####
    summed_up_reward_default = 0
    summed_up_reward_optimal = 0
    summed_up_reward_default_weighted = 0
    summed_up_reward_optimal_weighted = 0
    gamma = 0.9
    
    # simulate at least one step
    for i in range(max(round(trajectory_length), 1)):
        ##### DEFAULT ACTION #####
        normalized_reward_default, last_observation_state_default = get_next_state_and_normalized_reward(
            last_observation_state=last_observation_state_default,
            action=default_action[0],
            maximum_number_of_objects=maximum_number_of_objects,
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size, task=task, task_type=task_type)
        summed_up_reward_default += normalized_reward_default
        summed_up_reward_default_weighted += normalized_reward_default * math.pow(gamma, i)
        
        ##### OPTIMAL ACTION #####
        # get optimal action of agent
        actions, _, _, _, _ = policy.get_action_and_value_and_forward_model_prediction(
            fm_network=fm_network,
            obs=torch.tensor(last_observation_state_optimal, device=device, dtype=torch.float32).clone().detach(),
            deterministic=True,
            logger=logger,
            position_predicting=position_predicting,
            maximum_number_of_objects=maximum_number_of_objects)
        
        normalized_reward_optimal, last_observation_state_optimal = get_next_state_and_normalized_reward(
            last_observation_state=last_observation_state_optimal,
            action=actions[0],
            maximum_number_of_objects=maximum_number_of_objects,
            observation_width=observation_width,
            observation_height=observation_height,
            agent_size=agent_size, task=task, task_type=task_type)
        summed_up_reward_optimal += normalized_reward_optimal
        summed_up_reward_optimal_weighted += normalized_reward_optimal * math.pow(gamma, i)
    
    # get a mean reward between 0 and 1
    if weighting:
        summed_up_reward_default_normalized = summed_up_reward_default_weighted / (max(round(trajectory_length), 1))
        summed_up_reward_optimal_normalized = summed_up_reward_optimal_weighted / (max(round(trajectory_length), 1))
    else:
        summed_up_reward_default_normalized = summed_up_reward_default / (max(round(trajectory_length), 1))
        summed_up_reward_optimal_normalized = summed_up_reward_optimal / (max(round(trajectory_length), 1))
    
    # distance between the two trajectories
    need_for_control = (max(summed_up_reward_default_normalized, summed_up_reward_optimal_normalized)) - (
        min(summed_up_reward_default_normalized, summed_up_reward_optimal_normalized))
    
    # need for control is high if the rewards are quite different
    return need_for_control, summed_up_reward_default_normalized, summed_up_reward_optimal_normalized


def normalize_rewards(task: str, absolute_reward) -> float:
    # normalize reward with MinMaxScaler
    if task == "dodge":
        # normalize reward with MinMaxScaler: (reward - min) / (max - min)
        # the maximum reward is 10 -> when no obstacles are in the area
        # the minimum reward is ~-100 -> when the agent is completely surrounded by obstacles
        # the minimum condition never happens, in general are most rewards between 0 and 10
        # to not have a normalization, where 99% of the numbers are similar
        # we choose to clip the smallest 1% -> which is a clipping from -100 to -3 -> clip to -3
        if absolute_reward < -3:
            absolute_reward = np.array([-3])
        elif absolute_reward > 10:
            raise ValueError("Reward should not be higher than 10.")
        # normalized_reward = (absolute_reward - (-3)) / (10 - (-3))
        # FIXME: try different normalization!!!
        # reward between 0 and 0.5
        normalized_reward = (((0.5 - 0) * (absolute_reward - (-3))) / (10 - (-3))) + 0
    elif task == "collect":
        # normalize reward with MinMaxScaler: (reward - min) / (max - min)
        # the maximum reward is ~350 -> when the agent is completely surrounded by coins
        # the minimum reward is ~0 -> when no coins are in the area
        # the maximum condition never happens, in general are most rewards between 0 and 60
        # to not have a normalization, where 99% of the numbers are similar
        # we choose to clip the highest 1% -> which is a clipping from 0 to 62 -> clip to 62
        if absolute_reward > 62:
            absolute_reward = 62
        # it is possible that the reward is slightly negative (e.g. -0.0000001), which breaks our normalization
        elif absolute_reward < 0:
            absolute_reward = 0
        # normalized_reward = (absolute_reward - 0) / (62 - 0)
        # FIXME: try different normalization!!!
        # reward between 0 and 0.5
        normalized_reward = (((1 - 0.5) * (absolute_reward - 0)) / (62 - 0)) + 0.5
    else:
        raise NotImplementedError("Task {} not implemented".format(task))
    
    return normalized_reward


def calculate_trajectory_length(observation_height: int, prediction_error: float) -> float:
    # calculate the trajectory lengths through the prediction error
    # we decide that the trajectory length is half the observation size of the environment
    # when the prediction error is 0
    return - (observation_height / 2) * prediction_error + observation_height / 2


def get_next_state_and_normalized_reward(last_observation_state: torch.Tensor, action: torch.Tensor,
                                         maximum_number_of_objects: int, observation_width: int,
                                         observation_height: int,
                                         agent_size: int, task: str, task_type: str) -> tuple[float, torch.Tensor]:
    if not last_observation_state.shape[1] == (observation_width + 2) * observation_height:
        raise ValueError(
            f"The given observation width {observation_width} and height {observation_height} "
            f"do not match the observation shape: {last_observation_state.shape}."
            f"The second observation shape element {last_observation_state.shape[1]} should be "
            f"(observation_width + 2) * observation_height = {(observation_width + 2) * observation_height}.")
    if not (task == "dodge" or task == "collect"):
        raise NotImplementedError(f"The task {task} is not supported.")
    if not (task_type == "obstacle" or task_type == "coin"):
        raise NotImplementedError(f"The task type {task_type} is not supported.")
    if not ((task == "dodge" and task_type == "obstacle") or (task == "collect" and task_type == "coin")):
        raise NotImplementedError(f"The task {task} and task type {task_type} combination is not supported.")
    
    # get positions of last observation
    last_observation = get_position_and_object_positions_of_observation(
        torch.tensor(last_observation_state, device=device), maximum_number_of_objects=maximum_number_of_objects,
        observation_width=observation_width, observation_height=observation_height, agent_size=agent_size)
    
    # remove already overlapping objects
    if task == "collect":
        collected_objects_of_last_state = get_collected_objects(observation_positions=last_observation,
                                                                agent_size=agent_size,
                                                                observation_width=observation_width)
        for index in range(2, len(last_observation[0]), 2):
            x_position = int(last_observation[0][index])
            y_position = int(last_observation[0][index + 1])
            for collected_object in collected_objects_of_last_state:
                if x_position == collected_object['x'] and y_position == collected_object['y']:
                    last_observation[0][index] = 0
                    last_observation[0][index + 1] = 0
    
    # get next positions for new step with action
    last_observation = get_next_position_observation_moonlander(
        observations=last_observation,
        actions=action,
        observation_width=observation_width,
        agent_size=agent_size)
    
    # get collected objects to calculate reward
    x_position_of_agent = int(
        min(max(agent_size, last_observation[0][0]), observation_width - agent_size + 1))
    y_position_of_agent = int(last_observation[0][1])
    collected_objects = get_collected_objects(observation_positions=last_observation, agent_size=agent_size,
                                              observation_width=observation_width)
    
    # determine state of new positions for reward calculation
    last_observation_state = np.expand_dims(
        get_observation_of_position_and_object_positions(agent_and_object_positions=last_observation,
                                                         observation_height=observation_height,
                                                         observation_width=observation_width,
                                                         agent_size=agent_size,
                                                         task=task).flatten().cpu().numpy(),
        axis=0)
    
    # calculate reward
    rewards, _ = calculate_gaussian_reward(
        state=last_observation_state.reshape(observation_height, observation_width + 2),
        collected_objects=collected_objects,
        agent_size=agent_size,
        task_type=task_type,
        x_position_of_agent=x_position_of_agent,
        y_position_of_agent=y_position_of_agent)
    
    # normalize reward
    normalized_reward = normalize_rewards(task=task, absolute_reward=rewards)
    
    return normalized_reward, last_observation_state
