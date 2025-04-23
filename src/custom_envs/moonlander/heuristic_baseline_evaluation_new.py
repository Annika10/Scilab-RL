import os
import random
import csv
import ast
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
from custom_envs import ROOT_DIR

from src.custom_envs.moonlander.meta_env_pretrained_with_soc_wrapper import SoCWrapperEnv


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
    current_number_of_collect_task_one_actions = np.zeros(n_envs, dtype="int")
    current_number_of_collect_task_two_actions = np.zeros(n_envs, dtype="int")
    
    counter = 0
    counter_without_switch = 0
    
    last_action = np.array([0])
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
            last_action = actions
            counter_without_switch = 0
        else:
            counter_without_switch += 1
        if actions == np.array([0]):
            current_number_of_collect_task_one_actions += 1
        elif actions == np.array([1]):
            current_number_of_collect_task_two_actions += 1
        counter += 1
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
                    ###
                    
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    ### added by me
                    current_number_of_collected_objects_task_one[i] = 0
                    current_number_of_collected_objects_task_two[i] = 0
                    current_number_of_switches[i] = 0
                    current_number_of_collect_task_one_actions[i] = 0
                    current_number_of_collect_task_two_actions[i] = 0
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
             episode_number_of_collect_task_one_actions,
             episode_number_of_collected_objects_task_two,
             episode_number_of_switches,
             episode_number_of_collect_task_one_actions,
             episode_number_of_collect_task_two_actions,
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
    # TODO: test on same data for NfC and percentage
    filename_collect_easy_0 = "collect_easy_object_list_30_times_40_0.csv"
    filename_collect_hard_0 = "collect_hard_object_list_30_times_40_0.csv"
    filename_collect_easy_1 = "collect_easy_object_list_30_times_40_1.csv"
    filename_collect_hard_1 = "collect_hard_object_list_30_times_40_1.csv"
    
    list_of_filenames = [filename_collect_easy_0, filename_collect_hard_0, filename_collect_easy_1,
                         filename_collect_hard_1]
    dict_of_filename_to_object_dict_list = {}
    for filename in list_of_filenames:
        list_of_object_dict_lists = []
        with open(ROOT_DIR / "moonlander" / filename, "r") as file:
            lines = csv.reader(file)
            for line in lines:
                # first element is index
                # second element is the object list
                # form string to list of dictionaries
                list_of_object_dict_lists.append(ast.literal_eval(line[1]))
        dict_of_filename_to_object_dict_list[filename] = list_of_object_dict_lists
    
    ### DEFINE BEFORE ###
    
    mode = "switch_per_percentage"  # "SoC_switching", "switch_per_percentage"
    percentage_pairs = [[0, 1], [0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7],
                        [0.35, 0.65], [0.4, 0.6], [0.45, 0.55], [0.5, 0.5], [0.55, 0.45], [0.6, 0.4], [0.65, 0.35],
                        [0.7, 0.3], [0.75, 0.25], [0.8, 0.2], [0.85, 0.15], [0.9, 0.1], [0.95, 0.05], [1, 0]]
    meta_env_name = "MetaEnv-pretrained-without-SoC-v0"
    render = False
    
    collect_task_one_best_model_name = "collect_gaussian_with_distance_easy_positions_07_04_2025_best_model"
    collect_task_two_best_model_name = "collect_gaussian_with_distance_hard_positions_31_03_2025_best_model"
    config_file_name_collect_task_one = "../../../tests/test_data/levels/eval_config_collect_easy.yaml"
    config_file_name_collect_task_two = "../../../tests/test_data/levels/eval_config_collect_hard.yaml"
    filename_collect_task_one = filename_collect_easy_0
    filename_collect_task_two = filename_collect_hard_1
    
    directory = ROOT_DIR / "logs"
    
    n_eval_episodes = 100
    ####################
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    register_custom_envs()
    
    # Initialise the environment
    env = gym.make(meta_env_name, render_mode="human",
                   collect_task_one_best_model_name=collect_task_one_best_model_name,
                   collect_task_two_best_model_name=collect_task_two_best_model_name,
                   config_file_name_collect_task_one=config_file_name_collect_task_one,
                   config_file_name_collect_task_two=config_file_name_collect_task_two,
                   list_of_object_dict_lists_collect_task_one=dict_of_filename_to_object_dict_list[
                       filename_collect_task_one],
                   list_of_object_dict_lists_collect_task_two=dict_of_filename_to_object_dict_list[
                       filename_collect_task_two],
                   input_noise_in_subtasks_on=False, consecutive_frames=5)
    
    if mode == "SoC_switching":
        env = SoCWrapperEnv(env=env)
        # FIXME: why does it disappear when applying the wrapper?
        env.render_mode = "human"
        filepath_for_storage = directory / f"{collect_task_one_best_model_name}_{collect_task_two_best_model_name}_soc_switching.csv"
    else:
        filepath_for_storage = directory / f"{collect_task_one_best_model_name}_{collect_task_two_best_model_name}_percentage_pairs.csv"
    
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
                         "Episode number of collect task two actions"])
    
    if mode == "switch_per_percentage":
        for percentage_pair in percentage_pairs:
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
        _, _ = evaluate_policy(
            env=env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=render,
            return_episode_rewards=False,
            filepath_for_storage=filepath_for_storage,
            name_of_heuristic="SoC switching")
