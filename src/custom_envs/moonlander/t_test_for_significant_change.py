import csv
import ast
from scipy import stats
from src.custom_envs import ROOT_DIR

collect_task_one_best_model_name = "collect_gaussian_with_distance_easy_positions_07_04_2025_best_model"
collect_task_two_best_model_name = "collect_gaussian_with_distance_hard_positions_31_03_2025_best_model"

directory = ROOT_DIR / "logs"

soc_switching_filepath_for_storage = directory / f"{collect_task_one_best_model_name}_{collect_task_two_best_model_name}_soc_switching.csv"
percentage_switching_filepath_fors_storage = directory / f"{collect_task_one_best_model_name}_{collect_task_two_best_model_name}_percentage_pairs.csv"

# load episode collected objects
with open(soc_switching_filepath_for_storage, "r") as file:
    lines = csv.DictReader(file)
    for line in lines:
        soc_switching_episode_number_of_collected_objects_task_one = line[
            "Episode number of collected objects task one"]
        soc_switching_episode_number_of_collected_objects_task_two = line[
            "Episode number of collected objects task two"]

# form string to list
soc_switching_episode_number_of_collected_objects_task_one = ast.literal_eval(
    soc_switching_episode_number_of_collected_objects_task_one)
soc_switching_episode_number_of_collected_objects_task_two = ast.literal_eval(
    soc_switching_episode_number_of_collected_objects_task_two)

# load episode collected objects
percentage_switching_episode_number_of_collected_objects_task_one_dict_str = {}
percentage_switching_episode_number_of_collected_objects_task_one_dict = {}
percentage_switching_episode_number_of_collected_objects_task_two_dict_str = {}
percentage_switching_episode_number_of_collected_objects_task_two_dict = {}
with open(percentage_switching_filepath_for_storage, "r") as file:
    lines = csv.DictReader(file)
    for line in lines:
        percentage_switching_episode_number_of_collected_objects_task_one_dict_str[line["Heuristic name"]] = line[
            "Episode number of collected objects task one"]
        percentage_switching_episode_number_of_collected_objects_task_two_dict_str[line["Heuristic name"]] = line[
            "Episode number of collected objects task two"]

# form string to list
for key, value in percentage_switching_episode_number_of_collected_objects_task_one_dict_str.items():
    percentage_switching_episode_number_of_collected_objects_task_one_dict[key] = ast.literal_eval(value)
for key, value in percentage_switching_episode_number_of_collected_objects_task_two_dict_str.items():
    percentage_switching_episode_number_of_collected_objects_task_two_dict[key] = ast.literal_eval(value)

for key, value in percentage_switching_episode_number_of_collected_objects_task_one_dict.items():
    t_stat, p_value = stats.ttest_ind(value, soc_switching_episode_number_of_collected_objects_task_one)
    
    if p_value < 0.05:
        print(
            f"Reject the null hypothesis; there is a significant difference between collected "
            f"objects in task one by percentage switching of {key} and soc switching.")
    else:
        print(
            f"FAIL to reject the null hypothesis; there is no significant difference between collected "
            f"objects in task one by percentage switching of {key} and soc switching")

for key, value in percentage_switching_episode_number_of_collected_objects_task_two_dict.items():
    t_stat, p_value = stats.ttest_ind(value, soc_switching_episode_number_of_collected_objects_task_two)
    
    if p_value < 0.05:
        print(
            f"Reject the null hypothesis; there is a significant difference between collected "
            f"objects in task two by percentage switching of {key} and soc switching.")
    else:
        print(
            f"FAIL to reject the null hypothesis; there is no significant difference between collected "
            f"objects in task two by percentage switching of {key} and soc switching")
