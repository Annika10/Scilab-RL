import argparse
import os
import random
import csv
import torch
import math
import gymnasium as gym
from src.custom_envs.register_envs import register_custom_envs
import warnings
from typing import Any, Callable, Optional, Union
import numpy as np
from PIL import Image
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from src.utils.animation_util import LiveAnimationPlot
from src.custom_envs import ROOT_DIR

from src.custom_envs.moonlander.meta_env_pretrained_with_soc_wrapper import SoCObsAndRewardWrapperEnv
from src.custom_envs.moonlander.meta_env_pretrained_with_soc_reward_only_wrapper import SoCRewardOnlyWrapperEnv


# USE EVALUATE POLICY OF STABLE BASELINES 3 WITHOUT A MODEL BUT A HEURISTIC
def evaluate_policy(
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        action_sequence: list[int] = None,
        filepath_for_storage: str = None,
        name_of_heuristic: str = None,
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
    episode_number_of_collect_task_one_actions = []
    episode_number_of_collect_task_two_actions = []
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
    # per episode one value
    current_number_of_collected_objects_task_one = np.zeros(n_envs, dtype="int")
    current_number_of_collected_objects_task_two = np.zeros(n_envs, dtype="int")
    current_number_of_switches = np.zeros(n_envs, dtype="int")
    current_number_of_collect_task_one_actions = np.zeros(n_envs, dtype="int")
    current_number_of_collect_task_two_actions = np.zeros(n_envs, dtype="int")
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
    
    ### added by me
    if render:
        if name_of_heuristic == "SoC switching":
            animation = LiveAnimationPlot(y_axis_labels=['rewards', 'SoC task one', 'SoC task two', 'action'],
                                          env=env)
        else:
            animation = LiveAnimationPlot(y_axis_labels=['rewards'], env=env)
    ###
    
    while (episode_counts < episode_count_targets).any():
        ### added by me
        if action_sequence:
            actions = np.array([action_sequence[counter]])
        elif name_of_heuristic == "switch_every_frame":
            if last_action == np.array([0]):
                actions = np.array([1])
            elif last_action == np.array([1]):
                actions = np.array([0])
        else:
            if observations[0][0] > observations[0][1]:
                actions = np.array([1])
            elif observations[0][0] < observations[0][1]:
                actions = np.array([0])
            else:
                actions = np.array([random.randint(0, 1)])
        ###
        
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
            current_number_of_collect_task_one_actions += 1
        elif actions == np.array([1]):
            current_number_of_collect_task_two_actions += 1
        counter += 1
        last_observation = new_observations
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
                    episode_number_of_collect_task_one_actions.append(current_number_of_collect_task_one_actions[i])
                    episode_number_of_collect_task_two_actions.append(current_number_of_collect_task_two_actions[i])
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
                    # per episode one value
                    current_number_of_collected_objects_task_one[i] = 0
                    current_number_of_collected_objects_task_two[i] = 0
                    current_number_of_switches[i] = 0
                    current_number_of_collect_task_one_actions[i] = 0
                    current_number_of_collect_task_two_actions[i] = 0
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
            
            # needed to show the matplotlib plot
            animation.x_data[0].append(counter)
            animation.y_data[0].append(rewards[0])
            
            if name_of_heuristic == "SoC switching":
                animation.x_data[1].append(counter)
                animation.y_data[1].append(observations[0][0])
                animation.x_data[2].append(counter)
                animation.y_data[2].append(observations[0][1])
                animation.x_data[3].append(counter)
                animation.y_data[3].append(actions)
            
            animation.start_animation()
            
            observation_for_rendering = env.envs[0].env.env.env.observation_for_rendering
            a_min = np.min(observation_for_rendering)
            a_max = np.max(observation_for_rendering)
            a_scaled = 255 * (observation_for_rendering - a_min) / (a_max - a_min)
            
            im = Image.fromarray(a_scaled).convert('RGB')
            im.save(f"states/state_{counter}.png")
    
    if render:
        animation.save_animation("animation")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    ### added by me
    mean_collected_objects_task_one = np.mean(episode_number_of_collected_objects_task_one)
    std_collected_objects_task_one = np.std(episode_number_of_collected_objects_task_one)
    mean_collected_objects_task_two = np.mean(episode_number_of_collected_objects_task_two)
    std_collected_objects_task_two = np.std(episode_number_of_collected_objects_task_two)
    mean_number_of_switches = np.mean(episode_number_of_switches)
    std_number_of_switches = np.std(episode_number_of_switches)
    mean_number_of_collect_task_one_actions = np.mean(episode_number_of_collect_task_one_actions)
    std_number_of_collect_task_one_actions = np.std(episode_number_of_collect_task_one_actions)
    mean_number_of_collect_task_two_actions = np.mean(episode_number_of_collect_task_two_actions)
    std_number_of_collect_task_two_actions = np.std(episode_number_of_collect_task_two_actions)
    # get mean per episode
    mean_episode_number_of_consecutive_actions_in_task_one = [np.mean(number_of_consecutive_actions_in_task_one) for
                                                              number_of_consecutive_actions_in_task_one in
                                                              episode_number_of_consecutive_actions_in_task_one]
    mean_number_of_consecutive_actions_in_task_one = np.mean(mean_episode_number_of_consecutive_actions_in_task_one)
    std_number_of_consecutive_actions_in_task_one = np.std(mean_episode_number_of_consecutive_actions_in_task_one)
    mean_episode_number_of_consecutive_actions_in_task_two = [np.mean(number_of_consecutive_actions_in_task_two) for
                                                              number_of_consecutive_actions_in_task_two in
                                                              episode_number_of_consecutive_actions_in_task_two]
    mean_number_of_consecutive_actions_in_task_two = np.mean(mean_episode_number_of_consecutive_actions_in_task_two)
    std_number_of_consecutive_actions_in_task_two = np.std(mean_episode_number_of_consecutive_actions_in_task_two)
    mean_episode_objects_visible_when_switching_task_one = [np.mean(objects_visible_when_switching_task_one) for
                                                            objects_visible_when_switching_task_one in
                                                            episode_objects_visible_when_switching_task_one]
    mean_objects_visible_when_switching_task_one = np.mean(mean_episode_objects_visible_when_switching_task_one)
    std_objects_visible_when_switching_task_one = np.std(mean_episode_objects_visible_when_switching_task_one)
    mean_episode_objects_visible_when_not_switching_task_one = [np.mean(objects_visible_when_not_switching_task_one) for
                                                                objects_visible_when_not_switching_task_one in
                                                                episode_objects_visible_when_not_switching_task_one]
    mean_objects_visible_when_not_switching_task_one = np.mean(mean_episode_objects_visible_when_not_switching_task_one)
    std_objects_visible_when_not_switching_task_one = np.std(mean_episode_objects_visible_when_not_switching_task_one)
    mean_episode_objects_visible_when_switching_task_two = [np.mean(objects_visible_when_switching_task_two) for
                                                            objects_visible_when_switching_task_two in
                                                            episode_objects_visible_when_switching_task_two]
    mean_objects_visible_when_switching_task_two = np.mean(mean_episode_objects_visible_when_switching_task_two)
    std_objects_visible_when_switching_task_two = np.std(mean_episode_objects_visible_when_switching_task_two)
    mean_episode_objects_visible_when_not_switching_task_two = [np.mean(objects_visible_when_not_switching_task_two) for
                                                                objects_visible_when_not_switching_task_two in
                                                                episode_objects_visible_when_not_switching_task_two]
    mean_objects_visible_when_not_switching_task_two = np.mean(mean_episode_objects_visible_when_not_switching_task_two)
    std_objects_visible_when_not_switching_task_two = np.std(mean_episode_objects_visible_when_not_switching_task_two)
    mean_episode_mean_distance_to_visible_objects_when_switching_task_one = [
        np.mean(mean_distance_to_visible_objects_when_switching_task_one) for
        mean_distance_to_visible_objects_when_switching_task_one in
        episode_mean_distance_to_visible_objects_when_switching_task_one]
    mean_distance_to_visible_objects_when_switching_task_one = np.mean(
        mean_episode_mean_distance_to_visible_objects_when_switching_task_one)
    std_distance_to_visible_objects_when_switching_task_one = np.std(
        mean_episode_mean_distance_to_visible_objects_when_switching_task_one)
    mean_episode_mean_distance_to_visible_objects_when_not_switching_task_one = [
        np.mean(mean_distance_to_visible_objects_when_not_switching_task_one) for
        mean_distance_to_visible_objects_when_not_switching_task_one in
        episode_mean_distance_to_visible_objects_when_not_switching_task_one]
    mean_distance_to_visible_objects_when_not_switching_task_one = np.mean(
        mean_episode_mean_distance_to_visible_objects_when_not_switching_task_one)
    std_distance_to_visible_objects_when_not_switching_task_one = np.std(
        mean_episode_mean_distance_to_visible_objects_when_not_switching_task_one)
    mean_episode_mean_distance_to_visible_objects_when_switching_task_two = [
        np.mean(mean_distance_to_visible_objects_when_switching_task_two) for
        mean_distance_to_visible_objects_when_switching_task_two in
        episode_mean_distance_to_visible_objects_when_switching_task_two]
    mean_distance_to_visible_objects_when_switching_task_two = np.mean(
        mean_episode_mean_distance_to_visible_objects_when_switching_task_two)
    std_distance_to_visible_objects_when_switching_task_two = np.std(
        mean_episode_mean_distance_to_visible_objects_when_switching_task_two)
    mean_episode_mean_distance_to_visible_objects_when_not_switching_task_two = [
        np.mean(mean_distance_to_visible_objects_when_not_switching_task_two) for
        mean_distance_to_visible_objects_when_not_switching_task_two in
        episode_mean_distance_to_visible_objects_when_not_switching_task_two]
    mean_distance_to_visible_objects_when_not_switching_task_two = np.mean(
        mean_episode_mean_distance_to_visible_objects_when_not_switching_task_two)
    std_distance_to_visible_objects_when_not_switching_task_two = np.std(
        mean_episode_mean_distance_to_visible_objects_when_not_switching_task_two)
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(
        f"Mean number of collected objects task one: {mean_collected_objects_task_one:.2f} +/- {std_collected_objects_task_one:.2f}")
    print(
        f"Mean number of collected objects task two: {mean_collected_objects_task_two:.2f} +/- {std_collected_objects_task_two:.2f}")
    print(f"Mean number of switches: {mean_number_of_switches:.2f} +/- {std_number_of_switches:.2f}")
    print(
        f"Mean number of collect task one actions: {mean_number_of_collect_task_one_actions:.2f} +/- {std_number_of_collect_task_one_actions:.2f}")
    print(
        f"Mean number of collect task two actions: {mean_number_of_collect_task_two_actions:.2f} +/- {std_number_of_collect_task_two_actions:.2f}")
    print(
        f"Mean number of consecutive actions in task one: {mean_number_of_consecutive_actions_in_task_one:.2f} +/- {std_number_of_consecutive_actions_in_task_one:.2f}")
    print(
        f"Mean number of consecutive actions in task two: {mean_number_of_consecutive_actions_in_task_two:.2f} +/- {std_number_of_consecutive_actions_in_task_two:.2f}")
    print(
        f"Mean objects visible when switching task one: {mean_objects_visible_when_switching_task_one:.2f} +/- {std_objects_visible_when_switching_task_one:.2f}")
    print(
        f"Mean objects visible when not switching task one: {mean_objects_visible_when_not_switching_task_one:.2f} +/- {std_objects_visible_when_not_switching_task_one:.2f}")
    print(
        f"Mean objects visible when switching task two: {mean_objects_visible_when_switching_task_two:.2f} +/- {std_objects_visible_when_switching_task_two:.2f}")
    print(
        f"Mean objects visible when not switching task two: {mean_objects_visible_when_not_switching_task_two:.2f} +/- {std_objects_visible_when_not_switching_task_two:.2f}")
    print(
        f"Mean distance to visible objects when switching task one: {mean_distance_to_visible_objects_when_switching_task_one:.2f} +/- {std_distance_to_visible_objects_when_switching_task_one:.2f}")
    print(
        f"Mean distance to visible objects when not switching task one: {mean_distance_to_visible_objects_when_not_switching_task_one:.2f} +/- {std_distance_to_visible_objects_when_not_switching_task_one:.2f}")
    print(
        f"Mean distance to visible objects when switching task two: {mean_distance_to_visible_objects_when_switching_task_two:.2f} +/- {std_distance_to_visible_objects_when_switching_task_two:.2f}")
    print(
        f"Mean distance to visible objects when not switching task two: {mean_distance_to_visible_objects_when_not_switching_task_two:.2f} +/- {std_distance_to_visible_objects_when_not_switching_task_two:.2f}")
    
    with open(filepath_for_storage, "a") as file:
        writer = csv.writer(file)
        writer.writerow(
            [name_of_heuristic,
             mean_reward, std_reward,
             mean_collected_objects_task_one, std_collected_objects_task_one,
             mean_collected_objects_task_two, std_collected_objects_task_two,
             mean_number_of_switches, std_number_of_switches,
             mean_number_of_collect_task_one_actions, std_number_of_collect_task_one_actions,
             mean_number_of_collect_task_two_actions, std_number_of_collect_task_two_actions,
             episode_rewards,
             episode_number_of_collected_objects_task_one,
             episode_number_of_collected_objects_task_two,
             episode_number_of_switches,
             episode_number_of_collect_task_one_actions,
             episode_number_of_collect_task_two_actions,
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
             episode_true_if_it_was_switched
             ])
    ###
    
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def generate_random_numbers(number_of_blocks, number_of_remaining_elements):
    # Step 1: Generate x-1 random numbers between 0 and total
    random_numbers = sorted([random.randint(0, number_of_remaining_elements) for _ in range(number_of_blocks - 1)])
    
    # Step 2: Add 0 and total to the list and sort it
    random_numbers = [0] + random_numbers + [number_of_remaining_elements]
    
    # Step 3: Calculate the differences between consecutive numbers
    result = [random_numbers[i + 1] - random_numbers[i] for i in range(len(random_numbers) - 1)]
    
    return result


def create_blocks(list_of_current_action_occurrence, min_length):
    blocks = []
    lengths_of_current_action_occurrence = len(list_of_current_action_occurrence)
    # // floor division (returns the integer part of the division)
    # minimum number of blocks = 1, maximum number of blocks = 5*x=235 --> x=47
    # get random number of possible number of blocks
    number_of_blocks = random.randint(1, lengths_of_current_action_occurrence // min_length)
    
    # make a list for each block with the minimum length (5)
    minimum_block_lengths = [min_length] * number_of_blocks
    
    # Distribute remaining elements (that are not in the minimum lengths blocks)
    number_of_remaining_elements = lengths_of_current_action_occurrence - (min_length * number_of_blocks)
    number_of_elements_that_have_to_be_added_to_each_block = generate_random_numbers(number_of_blocks=number_of_blocks,
                                                                                     number_of_remaining_elements=number_of_remaining_elements)
    actual_block_lengths = [a + b for a, b in
                            zip(minimum_block_lengths, number_of_elements_that_have_to_be_added_to_each_block)]
    
    # Create blocks
    index = 0
    for length in actual_block_lengths:
        blocks.append(list_of_current_action_occurrence[index:index + length])
        index += length
    return blocks


def calculate_action_sequence_for_switch_per_percentage(dodge_percentage: float, collect_percentage: float,
                                                        n_eval_episodes: int) -> list[int]:
    action_sequence = []
    # in total, we need ~470/5=94 steps in one episode
    for i in range(n_eval_episodes):
        current_action_sequence = [0] * math.ceil(dodge_percentage * 94) + [1] * math.ceil(
            collect_percentage * 94)
        random.shuffle(current_action_sequence)
        action_sequence = action_sequence + current_action_sequence
    
    return action_sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--task_one_difficulty", type=str, default=None)
    parser.add_argument("--task_two_difficulty", type=str, default=None)
    parser.add_argument("--input_noise_string", type=str, default=None)
    args = parser.parse_args()
    
    mode = args.mode
    task_one_difficulty = args.task_one_difficulty
    task_two_difficulty = args.task_two_difficulty
    if args.input_noise_string:
        input_noise_string = "-" + args.input_noise_string
    else:
        input_noise_string = ""
    ### DEFINE BEFORE ###
    
    # mode = "switch_every_frame"  # "SoC_switching", "switch_per_percentage", "switch_every_frame"
    percentage_pairs = [[0, 1], [0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7],
                        [0.35, 0.65], [0.4, 0.6], [0.45, 0.55], [0.5, 0.5], [0.55, 0.45], [0.6, 0.4], [0.65, 0.35],
                        [0.7, 0.3], [0.75, 0.25], [0.8, 0.2], [0.85, 0.15], [0.9, 0.1], [0.95, 0.05], [1, 0]]
    # task_one_difficulty = "easy"  # "easy", "hard"
    # task_two_difficulty = "hard"  # "easy", "hard"
    # input_noise_string = "-input-noise"  # "", "-input-noise-in-one-task", "-input-noise-in-easy-task", "-input-noise-in-hard-task"
    
    render = False
    n_eval_episodes = 100
    
    ####################
    
    meta_env_name = f"MetaEnv-pretrained-without-SoC-{task_one_difficulty}-{task_two_difficulty}{input_noise_string}-v0"
    
    filename_collect_task_one = f"collect_{task_one_difficulty}_object_list_30_times_40_0.csv"
    filename_collect_task_two = f"collect_{task_two_difficulty}_object_list_30_times_40_1.csv"
    
    config_file_name_collect_task_one = f"../../../tests/test_data/levels/eval_config_collect_{task_one_difficulty}.yaml"
    config_file_name_collect_task_two = f"../../../tests/test_data/levels/eval_config_collect_{task_two_difficulty}.yaml"
    
    directory = ROOT_DIR / "logs"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    register_custom_envs()
    
    if mode == "SoC_switching":
        filepath_for_storage = directory / f"collect_{task_one_difficulty}_{task_two_difficulty}{input_noise_string}_soc_switching.csv"
    elif mode == "switch_per_percentage":
        filepath_for_storage = directory / f"collect_{task_one_difficulty}_{task_two_difficulty}{input_noise_string}_percentage_pairs.csv"
    else:
        filepath_for_storage = directory / f"collect_{task_one_difficulty}_{task_two_difficulty}{input_noise_string}_switch_every_frame.csv"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    # header for csv
    # write the objects list of each episode to file
    with open(filepath_for_storage, "a") as file:
        writer = csv.writer(file)
        writer.writerow(["Heuristic name", "Mean reward", "Std reward", "Mean number of collected objects task one",
                         "Std number of collected objects task one", "Mean number of collected objects task two",
                         "Std number of collected objects task two", "Mean number of switches",
                         "Std number of switches",
                         "Mean number of collect task one actions", "Std number of collect task one actions",
                         "Mean number of collect task two actions", "Std number of collect task two actions",
                         "Episode rewards",
                         "Episode number of collected objects task one", "Episode number of collected objects task two",
                         "Episode number of switches", "Episode number of collect task one actions",
                         "Episode number of collect task two actions",
                         "Episode number of consecutive actions in task one",
                         "Episode number of consecutive actions in task two",
                         "Episode objects visible when switching task one",
                         "Episode objects visible when not switching task one",
                         "Episode objects visible when switching task two",
                         "Episode objects visible when not switching task two",
                         "Episode mean distance to visible objects when switching task one",
                         "Episode mean distance to visible objects when not switching task one",
                         "Episode mean distance to visible objects when switching task two",
                         "Episode mean distance to visible objects when not switching task two",
                         "Episode true if it was switched"])
    
    if mode == "switch_per_percentage":
        for percentage_pair in percentage_pairs:
            # Initialise the environment
            env = gym.make(meta_env_name, render_mode="human",
                           config_file_name_collect_task_one=config_file_name_collect_task_one,
                           config_file_name_collect_task_two=config_file_name_collect_task_two,
                           list_of_object_dict_lists_collect_task_one_filename=filename_collect_task_one,
                           list_of_object_dict_lists_collect_task_two_filename=filename_collect_task_two)
            env = SoCRewardOnlyWrapperEnv(env=env)
            # FIXME: why does it disappear when applying the wrapper?
            env.render_mode = "human"
            print(
                f"Currently evaluating collect task one percentage: {percentage_pair[0]}"
                f" and collect task two percentage: {percentage_pair[1]}")
            action_sequence = calculate_action_sequence_for_switch_per_percentage(
                dodge_percentage=percentage_pair[0], collect_percentage=percentage_pair[1],
                n_eval_episodes=n_eval_episodes)
            _, _ = evaluate_policy(
                env=env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=render,
                return_episode_rewards=False,
                action_sequence=action_sequence,
                filepath_for_storage=filepath_for_storage,
                name_of_heuristic=str(percentage_pair))
    
    elif mode == "SoC_switching":
        # Initialise the environment
        env = gym.make(meta_env_name, render_mode="human",
                       config_file_name_collect_task_one=config_file_name_collect_task_one,
                       config_file_name_collect_task_two=config_file_name_collect_task_two,
                       list_of_object_dict_lists_collect_task_one_filename=filename_collect_task_one,
                       list_of_object_dict_lists_collect_task_two_filename=filename_collect_task_two)
        env = SoCRewardOnlyWrapperEnv(env=env)
        env = SoCObsAndRewardWrapperEnv(env=env)
        # FIXME: why does it disappear when applying the wrapper?
        env.render_mode = "human"
        
        _, _ = evaluate_policy(
            env=env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=render,
            return_episode_rewards=False,
            filepath_for_storage=filepath_for_storage,
            name_of_heuristic="SoC switching")
    
    elif mode == "switch_every_frame":
        # Initialise the environment
        env = gym.make(meta_env_name, render_mode="human",
                       config_file_name_collect_task_one=config_file_name_collect_task_one,
                       config_file_name_collect_task_two=config_file_name_collect_task_two,
                       list_of_object_dict_lists_collect_task_one_filename=filename_collect_task_one,
                       list_of_object_dict_lists_collect_task_two_filename=filename_collect_task_two)
        env = SoCRewardOnlyWrapperEnv(env=env)
        # FIXME: why does it disappear when applying the wrapper?
        env.render_mode = "human"
        
        _, _ = evaluate_policy(
            env=env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=render,
            return_episode_rewards=False,
            filepath_for_storage=filepath_for_storage,
            name_of_heuristic="switch_every_frame")
