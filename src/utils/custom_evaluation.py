import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import gymnasium as gym
import numpy as np
import torch
import pandas as pd
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from src.custom_envs.moonlander.utils import get_position_and_object_positions_of_observation, \
    get_observation_of_position_and_object_positions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# modified copy from stable baselines
def evaluate_policy_moonlander(
        model: "type_aliases.PolicyPredictor",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor
    
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
    
    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    
    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
    
    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    ### added by me
    episode_number_of_crashed_or_collected_objects = []
    ###
    
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    ### added by me
    current_number_of_crashed_or_collected_objects = np.zeros(n_envs, dtype="int")
    ###
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        ### added by me
        current_number_of_crashed_or_collected_objects += infos[0]["number_of_crashed_or_collected_objects"]
        ###
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                
                if callback is not None:
                    callback(locals(), globals())
                
                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    
                    ### added by me
                    episode_number_of_crashed_or_collected_objects.append(
                        current_number_of_crashed_or_collected_objects[i])
                    ###
                    
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    ### added by me
                    current_number_of_crashed_or_collected_objects[i] = 0
                    ###
        
        observations = new_observations
        
        if render:
            env.render()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_number_of_crashed_or_collected_objects
    return mean_reward, std_reward


def evaluate_policy_meta_agent_new(
        model: "type_aliases.PolicyPredictor",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        ### added by me
        logger=None,
        ###
) -> Union[tuple[float, float], tuple[list[float]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor
    
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
    
    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    
    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
    
    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    ### added by me
    episode_number_of_collected_objects_task_one = []
    episode_number_of_collected_objects_task_two = []
    episode_number_of_switches = []
    episode_number_of_task_one_actions = []
    episode_number_of_task_two_actions = []
    # list in list
    episode_number_of_consecutive_actions_in_task_one = []
    episode_number_of_consecutive_actions_in_task_two = []
    episode_objects_visible_when_switching_task_one = []
    episode_objects_visible_when_not_switching_task_one = []
    episode_objects_visible_when_switching_task_two = []
    episode_objects_visible_when_not_switching_task_two = []
    episode_mean_distance_to_visible_objects_when_switching_task_one = []
    episode_mean_distance_to_visible_objects_when_not_switching_task_one = []
    episode_mean_distance_to_visible_objects_when_switching_task_two = []
    episode_mean_distance_to_visible_objects_when_not_switching_task_two = []
    # dict in list
    episode_true_if_it_was_switched = []
    ###
    
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    ### added by me
    current_number_of_collected_objects_task_one = np.zeros(n_envs, dtype="int")
    current_number_of_collected_objects_task_two = np.zeros(n_envs, dtype="int")
    current_number_of_switches = np.zeros(n_envs, dtype="int")
    current_number_of_task_one_actions = np.zeros(n_envs, dtype="int")
    current_number_of_task_two_actions = np.zeros(n_envs, dtype="int")
    # per episode one list
    current_number_of_consecutive_actions_in_task_one = []
    current_number_of_consecutive_actions_in_task_two = []
    current_objects_visible_when_switching_task_one = []
    current_objects_visible_when_not_switching_task_one = []
    current_objects_visible_when_switching_task_two = []
    current_objects_visible_when_not_switching_task_two = []
    current_mean_distance_to_visible_objects_when_switching_task_one = []
    current_mean_distance_to_visible_objects_when_not_switching_task_one = []
    current_mean_distance_to_visible_objects_when_switching_task_two = []
    current_mean_distance_to_visible_objects_when_not_switching_task_two = []
    # dict
    current_true_if_it_was_switched = {}
    
    counter = 0
    counter_without_switch = 0
    
    last_action = np.array([0])
    last_observation = np.zeros((1, 44))
    ###
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        ### added by me
        current_number_of_collected_objects_task_one += infos[0]["collect_task_one_collected_objects"]
        current_number_of_collected_objects_task_two += infos[0]["collect_task_two_collected_objects"]
        
        if not (last_action == actions).item():
            current_number_of_switches += 1
            current_true_if_it_was_switched[counter] = True
            
            # use last observation to see what was happening before the switch
            if actions == np.array([0]):
                # last observation was with action one
                # objects of task two:
                objects_of_task_two = last_observation[0, 24:]
                # count objects
                current_objects_visible_when_switching_task_two.append(int(np.count_nonzero(objects_of_task_two) / 2))
                
                distances_to_agent = []
                for i in range(0, len(objects_of_task_two), 2):
                    coordinate_of_agent = last_observation[0, 22:24]
                    current_coordinate_of_object = objects_of_task_two[i:i + 2]
                    if not (current_coordinate_of_object == np.array([0, 0])).all():
                        distances_to_agent.append(math.dist(coordinate_of_agent, current_coordinate_of_object))
                
                if len(distances_to_agent) > 0:
                    current_mean_distance_to_visible_objects_when_switching_task_two.append(
                        sum(distances_to_agent) / len(distances_to_agent))
                
                # save from last action how many actions were done without switching
                current_number_of_consecutive_actions_in_task_two.append(counter_without_switch + 1)
            elif actions == np.array([1]):
                # last observation was with action zero
                # objects of task one:
                objects_of_task_one = last_observation[0, 2:22]
                # count objects
                current_objects_visible_when_switching_task_one.append(int(np.count_nonzero(objects_of_task_one) / 2))
                
                distances_to_agent = []
                for i in range(0, len(objects_of_task_one), 2):
                    coordinate_of_agent = last_observation[0, 0:2]
                    current_coordinate_of_object = objects_of_task_one[i:i + 2]
                    if not (current_coordinate_of_object == np.array([0, 0])).all():
                        distances_to_agent.append(math.dist(coordinate_of_agent, current_coordinate_of_object))
                
                if len(distances_to_agent) > 0:
                    current_mean_distance_to_visible_objects_when_switching_task_one.append(
                        sum(distances_to_agent) / len(distances_to_agent))
                
                # save from last action how many actions were done without switching
                current_number_of_consecutive_actions_in_task_one.append(counter_without_switch + 1)
            
            last_action = actions
            counter_without_switch = 0
        else:
            current_true_if_it_was_switched[counter] = False
            counter_without_switch += 1
            
            if actions == np.array([0]):
                # last observation was with action zero
                # objects of task one:
                objects_of_task_one = last_observation[0, 2:22]
                # count objects
                current_objects_visible_when_not_switching_task_one.append(
                    int(np.count_nonzero(objects_of_task_one) / 2))
                
                distances_to_agent = []
                for i in range(0, len(objects_of_task_one), 2):
                    coordinate_of_agent = last_observation[0, 0:2]
                    current_coordinate_of_object = objects_of_task_one[i:i + 2]
                    if not (current_coordinate_of_object == np.array([0, 0])).all():
                        distances_to_agent.append(math.dist(coordinate_of_agent, current_coordinate_of_object))
                
                if len(distances_to_agent) > 0:
                    current_mean_distance_to_visible_objects_when_not_switching_task_one.append(
                        sum(distances_to_agent) / len(distances_to_agent))
            
            elif actions == np.array([1]):
                # last observation was with action one
                # objects of task two
                objects_of_task_two = last_observation[0, 24:]
                # count objects
                current_objects_visible_when_not_switching_task_two.append(
                    int(np.count_nonzero(objects_of_task_two) / 2))
                
                distances_to_agent = []
                for i in range(0, len(objects_of_task_two), 2):
                    coordinate_of_agent = last_observation[0, 22:24]
                    current_coordinate_of_object = objects_of_task_one[i:i + 2]
                    if not (current_coordinate_of_object == np.array([0, 0])).all():
                        distances_to_agent.append(math.dist(coordinate_of_agent, current_coordinate_of_object))
                
                if len(distances_to_agent) > 0:
                    current_mean_distance_to_visible_objects_when_not_switching_task_two.append(
                        sum(distances_to_agent) / len(distances_to_agent))
        
        if actions == np.array([0]):
            current_number_of_task_one_actions += 1
        elif actions == np.array([1]):
            current_number_of_task_two_actions += 1
        counter += 1
        last_observation = np.expand_dims(infos[0]["position_state"], axis=0)
        
        logger.record("eval/SoC_collect_task_one", new_observations[0][0])
        logger.record("eval/SoC_collect_task_two", new_observations[0][1])
        # when using SoC wrapper
        if "prediction_error" in infos[0]:
            logger.record("eval/prediction_error", infos[0]["prediction_error"])
        if "need_for_control" in infos[0]:
            logger.record("eval/need_for_control", infos[0]["need_for_control"])
        ###
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                
                if callback is not None:
                    callback(locals(), globals())
                
                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    
                    ### added by me
                    episode_number_of_collected_objects_task_one.append(current_number_of_collected_objects_task_one[i])
                    episode_number_of_collected_objects_task_two.append(current_number_of_collected_objects_task_two[i])
                    episode_number_of_switches.append(current_number_of_switches[i])
                    episode_number_of_task_one_actions.append(current_number_of_task_one_actions[i])
                    episode_number_of_task_two_actions.append(current_number_of_task_two_actions[i])
                    episode_number_of_consecutive_actions_in_task_one.append(
                        current_number_of_consecutive_actions_in_task_one)
                    episode_number_of_consecutive_actions_in_task_two.append(
                        current_number_of_consecutive_actions_in_task_two)
                    episode_objects_visible_when_switching_task_one.append(
                        current_objects_visible_when_switching_task_one)
                    episode_objects_visible_when_not_switching_task_one.append(
                        current_objects_visible_when_not_switching_task_one)
                    episode_objects_visible_when_switching_task_two.append(
                        current_objects_visible_when_switching_task_two)
                    episode_objects_visible_when_not_switching_task_two.append(
                        current_objects_visible_when_not_switching_task_two)
                    episode_mean_distance_to_visible_objects_when_switching_task_one.append(
                        current_mean_distance_to_visible_objects_when_switching_task_one)
                    episode_mean_distance_to_visible_objects_when_not_switching_task_one.append(
                        current_mean_distance_to_visible_objects_when_not_switching_task_one)
                    episode_mean_distance_to_visible_objects_when_switching_task_two.append(
                        current_mean_distance_to_visible_objects_when_switching_task_two)
                    episode_mean_distance_to_visible_objects_when_not_switching_task_two.append(
                        current_mean_distance_to_visible_objects_when_not_switching_task_two)
                    episode_true_if_it_was_switched.append(current_true_if_it_was_switched)
                    ###
                    
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    ### added by me
                    current_number_of_collected_objects_task_one[i] = 0
                    current_number_of_collected_objects_task_two[i] = 0
                    current_number_of_switches[i] = 0
                    current_number_of_task_one_actions[i] = 0
                    current_number_of_task_two_actions[i] = 0
                    # per episode one list
                    current_number_of_consecutive_actions_in_task_one = []
                    current_number_of_consecutive_actions_in_task_two = []
                    current_objects_visible_when_switching_task_one = []
                    current_objects_visible_when_not_switching_task_one = []
                    current_objects_visible_when_switching_task_two = []
                    current_objects_visible_when_not_switching_task_two = []
                    current_mean_distance_to_visible_objects_when_switching_task_one = []
                    current_mean_distance_to_visible_objects_when_not_switching_task_one = []
                    current_mean_distance_to_visible_objects_when_switching_task_two = []
                    current_mean_distance_to_visible_objects_when_not_switching_task_two = []
                    # dict
                    current_true_if_it_was_switched = {}
                    ###
        
        observations = new_observations
        
        if render:
            env.render()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return (episode_rewards,
                episode_lengths,
                episode_number_of_collected_objects_task_one,
                episode_number_of_collected_objects_task_two,
                episode_number_of_switches,
                episode_number_of_task_one_actions,
                episode_number_of_task_two_actions,
                episode_number_of_consecutive_actions_in_task_one,
                episode_number_of_consecutive_actions_in_task_two,
                episode_objects_visible_when_switching_task_one,
                episode_objects_visible_when_not_switching_task_one,
                episode_objects_visible_when_switching_task_two,
                episode_objects_visible_when_not_switching_task_two,
                episode_mean_distance_to_visible_objects_when_switching_task_one,
                episode_mean_distance_to_visible_objects_when_not_switching_task_one,
                episode_mean_distance_to_visible_objects_when_switching_task_two,
                episode_mean_distance_to_visible_objects_when_not_switching_task_two,
                episode_true_if_it_was_switched)
    return mean_reward, std_reward


def evaluate_policy_meta_agent(
        model: "type_aliases.PolicyPredictor",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        # from us
        callback_metric_viz=None,
        logger=None,
        counter=None,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    From the stable-baselines3 evaluation implementation.

    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :param counter: Counter to name csv files for analysis
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor
    
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
    
    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    
    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
    
    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    first_step = True
    
    dict_avoid = {"timesteps": [], "player_pos": [], "active_task": [], "input_noise": [], "current_reward": [],
                  "list_of_visible_objects": [], "number_of_visible_objects": [], "distance_to_closest_object": []}
    
    dict_collect = {"timesteps": [], "player_pos": [], "active_task": [], "input_noise": [], "current_reward": [],
                    "list_of_visible_objects": [], "number_of_visible_objects": [], "distance_to_closest_object": []}
    
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    
    duration = 0
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    ### from me
    last_action = np.array([0])
    current_number_of_crashed_objects = np.zeros(n_envs, dtype="int")
    current_number_of_collected_objects = np.zeros(n_envs, dtype="int")
    current_number_of_switches = np.zeros(n_envs, dtype="int")
    current_number_of_dodge_actions = np.zeros(n_envs, dtype="int")
    current_number_of_collect_actions = np.zeros(n_envs, dtype="int")
    episode_number_of_crashed_objects = []
    episode_number_of_collected_objects = []
    episode_number_of_switches = []
    episode_number_of_dodge_actions = []
    episode_number_of_collect_actions = []
    ###
    
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        
        # ### evaluate the forward model
        # observation_width = env.env_method("get_wrapper_attr", "observation_width")[0]
        # observation_height = env.env_method("get_wrapper_attr", "observation_height")[0]
        # agent_size = env.env_method("get_wrapper_attr", "agent_size")[0]
        # maximum_number_of_objects = env.env_method("get_wrapper_attr", "maximum_number_of_objects")[0]
        #
        # last_observations = np.array([])
        # if actions[0] == 0:
        #     for i in range(0, observation_height * 2, 2):
        #         last_observations = np.concatenate(
        #             (last_observations, observations[0][i * (observation_width + 2):(i + 1) * (observation_width + 2)]),
        #             axis=0)
        #     trained_agent = env.env_method("get_wrapper_attr", "trained_dodge_asteroids")[0]
        #     task = "dodge"
        # else:
        #     for i in range(1, observation_height * 2 + 1, 2):
        #         last_observations = np.concatenate(
        #             (last_observations, observations[0][i * (observation_width + 2):(i + 1) * (observation_width + 2)]),
        #             axis=0)
        #     trained_agent = env.env_method("get_wrapper_attr", "trained_collect_asteroids")[0]
        #     task = "collect"
        #
        # last_observations = np.expand_dims(last_observations, axis=0)
        # last_observations = get_position_and_object_positions_of_observation(
        #     obs=torch.tensor(last_observations, device=device),
        #     maximum_number_of_objects=maximum_number_of_objects,
        #     observation_width=observation_width,
        #     observation_height=observation_height,
        #     agent_size=agent_size)
        #
        # forward_model = trained_agent.fm_network
        # # tensor (1, 22) & tensor (1, 1)
        # forward_model_prediction = forward_model(last_observations, torch.tensor([actions]).float())
        
        # belief_state = get_observation_of_position_and_object_positions(agent_and_object_positions=
        #                                                                 forward_model_prediction.mean[
        #                                                                     0][:-1].cpu().unsqueeze(0),
        #                                                                 observation_height=30,
        #                                                                 observation_width=40,
        #                                                                 agent_size=2,
        #                                                                 task=task).flatten().cpu().numpy()
        # belief_state = np.expand_dims(belief_state, 0)
        new_observations, rewards, dones, infos = env.step(actions)
        
        # new_observations_current_task = np.array([])
        # if actions[0] == 0:
        #     for i in range(0, observation_height * 2, 2):
        #         new_observations_current_task = np.concatenate(
        #             (new_observations_current_task,
        #              new_observations[0][i * (observation_width + 2):(i + 1) * (observation_width + 2)]),
        #             axis=0)
        # else:
        #     for i in range(1, observation_height * 2 + 1, 2):
        #         new_observations_current_task = np.concatenate(
        #             (new_observations_current_task,
        #              new_observations[0][i * (observation_width + 2):(i + 1) * (observation_width + 2)]),
        #             axis=0)
        #
        # new_observations_current_task = np.expand_dims(new_observations_current_task, axis=0)
        # positions_of_new_observation = \
        #     get_position_and_object_positions_of_observation(
        #         obs=torch.tensor(new_observations_current_task, device=device),
        #         # fixme: not hardcoded
        #         maximum_number_of_objects=10,
        #         observation_width=observation_width,
        #         observation_height=observation_height,
        #         agent_size=agent_size)
        
        # print("forward_model_prediction", torch.round(forward_model_prediction.mean))
        # print("positions_of_new_observation", positions_of_new_observation)
        
        ### custom code
        
        info_dict = infos[0]
        
        noise = info_dict['input_noise']
        
        if first_step:
            difficulty_dodge = info_dict['difficulty_dodge']
            difficulty_collect = info_dict['difficulty_collect']
            if noise == 0:
                input_noise = 'no'
            else:
                input_noise = 'yes'
            first_step = False
        
        list_of_visible_objects_dodge = []
        list_of_visible_objects_collect = []
        
        for i in range(1, info_dict["objects_dodge"].size(dim=1), 2):
            x = info_dict["objects_dodge"][0, i - 1]
            y = info_dict["objects_dodge"][0, i]
            temp_list = [x.item(), y.item()]
            list_of_visible_objects_dodge.append(temp_list)
        
        for i in range(1, info_dict["objects_collect"].size(dim=1), 2):
            x = info_dict["objects_collect"][0, i - 1]
            y = info_dict["objects_collect"][0, i]
            temp_list = [x.item(), y.item()]
            list_of_visible_objects_collect.append(temp_list)
        
        # remove player position
        player_dodge = list_of_visible_objects_dodge.pop(0)
        player_collect = list_of_visible_objects_collect.pop(0)
        
        # remove entrys without objects
        list_of_visible_objects_dodge = [i for i in list_of_visible_objects_dodge if i != [0.0, 0.0]]
        list_of_visible_objects_collect = [i for i in list_of_visible_objects_collect if i != [0.0, 0.0]]
        
        distances_dodge = []
        distance_to_closest_object_dodge = 0
        for object in list_of_visible_objects_dodge:
            distances_dodge.append(np.linalg.norm(np.array(player_dodge) - np.array(object)))
        if distances_dodge:
            distance_to_closest_object_dodge = min(distances_dodge)
        else:
            distance_to_closest_object_dodge = np.nan
        
        distances_collect = []
        distance_to_closest_object_collect = 0
        for object in list_of_visible_objects_collect:
            distances_collect.append(np.linalg.norm(np.array(player_collect) - np.array(object)))
        if distances_collect:
            distance_to_closest_object_collect = min(distances_collect)
        else:
            distance_to_closest_object_collect = np.nan
        
        dict_avoid["timesteps"].append(duration)
        dict_collect["timesteps"].append(duration)
        
        duration = duration + 1
        
        dict_avoid["player_pos"].append(info_dict['dodge_position_before'])
        dict_collect["player_pos"].append(info_dict['collect_position_before'])
        
        dict_avoid["input_noise"].append(noise)
        dict_collect["input_noise"].append(noise)
        
        dict_avoid["current_reward"].append(info_dict['reward_dodge'])
        dict_collect["current_reward"].append(info_dict['reward_collect'])
        
        if info_dict['action_meta'] == 0:
            active_task = True
        else:
            active_task = False
        dict_avoid["active_task"].append(active_task)
        dict_collect["active_task"].append(not active_task)
        
        dict_avoid["list_of_visible_objects"].append(list_of_visible_objects_dodge)
        dict_collect["list_of_visible_objects"].append(list_of_visible_objects_collect)
        
        dict_avoid["number_of_visible_objects"].append(len(list_of_visible_objects_dodge))
        dict_collect["number_of_visible_objects"].append(len(list_of_visible_objects_collect))
        
        dict_avoid["distance_to_closest_object"].append(distance_to_closest_object_dodge)
        dict_collect["distance_to_closest_object"].append(distance_to_closest_object_collect)
        
        logger.record("eval/action_meta", info_dict["action_meta"])
        logger.record("eval/dodge_position_before", info_dict["dodge_position_before"])
        logger.record("eval/collect_position_before", info_dict["collect_position_before"])
        logger.record("eval/dodge_action", info_dict["dodge_action"])
        logger.record("eval/collect_action", info_dict["collect_action"])
        logger.record("eval/input_noise", info_dict["input_noise"])
        logger.record("eval/dodge_next_position", info_dict["dodge_next_position"])
        logger.record("eval/collect_next_position", info_dict["collect_next_position"])
        logger.record("eval/predicted_dodge_next_position", info_dict["predicted_dodge_next_position"])
        logger.record("eval/predicted_collect_next_position", info_dict["predicted_collect_next_position"])
        logger.record("eval/prediction_error_dodge", info_dict["prediction_error_dodge"])
        logger.record("eval/prediction_error_collect", info_dict["prediction_error_collect"])
        logger.record("eval/need_for_control_dodge", info_dict["need_for_control_dodge"])
        logger.record("eval/need_for_control_collect", info_dict["need_for_control_collect"])
        logger.record("eval/SoC_dodge", info_dict["SoC_dodge"])
        logger.record("eval/SoC_collect", info_dict["SoC_collect"])
        logger.record("eval/reward_dodge", info_dict["reward_dodge"])
        logger.record("eval/reward_collect", info_dict["reward_collect"])
        logger.record("eval/dodge_object_crashed",
                      info_dict["info_dodge"][0]["number_of_crashed_or_collected_objects"])
        logger.record("eval/collect_object_collected",
                      info_dict["info_collect"][0]["number_of_crashed_or_collected_objects"])
        logger.record("eval/rollout_rewards_step", float(rewards.mean()))
        logger.record_mean("eval/rollout_rewards_mean", float(rewards.mean()))
        
        # dodge/collect env
        if "simple" in infos[0].keys():
            logger.record("eval/number_of_crashed_or_collected_objects",
                          float(infos[0]["number_of_crashed_or_collected_objects"]))
        # trigger metric visualization
        if callback_metric_viz:
            callback_metric_viz._on_step()
        
        current_number_of_crashed_objects += info_dict["info_dodge"][0]["number_of_crashed_or_collected_objects"]
        current_number_of_collected_objects += info_dict["info_collect"][0]["number_of_crashed_or_collected_objects"]
        if not (last_action == actions).item():
            current_number_of_switches += 1
            last_action = actions
        if actions == np.array([0]):
            current_number_of_dodge_actions += 1
        elif actions == np.array([1]):
            current_number_of_collect_actions += 1
        
        ### until here
        
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                
                if callback is not None:
                    callback(locals(), globals())
                
                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                        
                        ### from me
                        episode_number_of_crashed_objects.append(current_number_of_crashed_objects[i])
                        episode_number_of_collected_objects.append(current_number_of_collected_objects[i])
                        episode_number_of_switches.append(current_number_of_switches[i])
                        episode_number_of_dodge_actions.append(current_number_of_dodge_actions[i])
                        episode_number_of_collect_actions.append(current_number_of_collect_actions[i])
                        ###
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    
                    ### from me
                    current_number_of_crashed_objects[i] = 0
                    current_number_of_collected_objects[i] = 0
                    current_number_of_switches[i] = 0
                    current_number_of_dodge_actions[i] = 0
                    current_number_of_collect_actions[i] = 0
        
        observations = new_observations
        
        if render:
            env.render()
    
    dir_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agent_data')
    
    if not os.path.isdir(dir_path):
        print("Creating directory for evaluation files of agent")
        os.mkdir(dir_path)
    
    avoid_df = pd.DataFrame.from_dict(data=dict_avoid)
    collect_df = pd.DataFrame.from_dict(data=dict_collect)
    avoid_df.to_csv(
        dir_path + '/agent_' + difficulty_dodge + '_' + difficulty_collect + '_' + input_noise + '_' + str(
            counter) + '_avoid.csv', index=False)
    collect_df.to_csv(
        dir_path + '/agent_' + difficulty_dodge + '_' + difficulty_collect + '_' + input_noise + '_' + str(
            counter) + '_collect.csv', index=False)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_number_of_crashed_objects, episode_number_of_collected_objects, episode_number_of_switches, episode_number_of_dodge_actions, episode_number_of_collect_actions
    return mean_reward, std_reward
