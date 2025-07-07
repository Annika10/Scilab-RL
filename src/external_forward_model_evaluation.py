import torch
from src.custom_algorithms.cleanppofm.forward_model import ProbabilisticForwardNetPositionPredictionIncludingReward
import gymnasium as gym
from src.custom_envs.register_envs import register_custom_envs
from src.custom_envs.moonlander.utils import get_next_position_observation_moonlander

observations = torch.tensor([[19., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [27., 1., 15., 3., 32., 5., 24., 8., 9., 24., 26., 29., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [2., 1., 37., 20., 38., 26., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [22., 1., 39., 0., 39., 4., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [33., 1., 15., 16., 24., 19., 4., 29., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [19., 1., 14., 16., 31., 22., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [22., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [5., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [25., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [25., 1., 31., 4., 32., 15., 33., 28., 30., 29., 31., 29., 32., 29.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [8., 1., 17., 28., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [25., 1., 7., 21., 20., 22., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [16., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [6., 1., 30., 0., 36., 20., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [4., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [22., 1., 8., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [12., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [5., 1., 30., 5., 36., 25., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [25., 1., 16., 27., 33., 29., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [27., 1., 14., 7., 25., 25., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [10., 1., 34., 2., 13., 4., 10., 8., 22., 8., 35., 9., 27., 11.,
                              11., 25., 0., 0., 0., 0., 0., 0.],
                             [10., 1., 35., 21., 35., 28., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [30., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [16., 1., 19., 10., 2., 11., 2., 12., 34., 28., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [6., 1., 31., 7., 5., 11., 27., 11., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [22., 1., 6., 8., 29., 10., 12., 16., 4., 23., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [12., 1., 36., 12., 17., 29., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [8., 1., 17., 13., 5., 18., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [29., 1., 39., 2., 14., 4., 6., 8., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [21., 1., 3., 9., 39., 22., 23., 29., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [34., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [4., 1., 37., 28., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [8., 1., 35., 1., 31., 4., 8., 23., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [28., 1., 14., 24., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [8., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [13., 1., 35., 25., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [2., 1., 11., 0., 31., 21., 5., 25., 27., 25., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [31., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [30., 1., 14., 22., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [4., 1., 31., 12., 5., 16., 27., 16., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [15., 1., 14., 18., 31., 24., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [16., 1., 34., 14., 30., 17., 6., 23., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [18., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [18., 1., 39., 17., 23., 27., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [33., 1., 15., 15., 24., 18., 4., 28., 38., 29., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [8., 1., 35., 18., 35., 25., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [14., 1., 32., 2., 33., 15., 30., 17., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [27., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [37., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [18., 1., 8., 4., 2., 29., 19., 29., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [27., 1., 14., 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [30., 1., 14., 18., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [33., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [28., 1., 38., 5., 5., 10., 36., 26., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [28., 1., 7., 8., 20., 9., 39., 26., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [8., 1., 35., 17., 35., 24., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [32., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [39., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [10., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [10., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [14., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [30., 1., 31., 29., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.],
                             [18., 1., 35., 7., 32., 15., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.]])

register_custom_envs()
fm_parameters = {'hidden_size': 256, 'learning_rate': 0.001, 'reward_eta': 0.2}
reward_predicting = True
maximum_number_of_objects = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('MoonlanderWorld-dodge-gaussian-v0', reward_function="gaussian")
env.reset()

fm_network = ProbabilisticForwardNetPositionPredictionIncludingReward(env, fm_parameters,
                                                                      maximum_number_of_objects=maximum_number_of_objects).to(
    device)

cleanppofm_model = torch.load(
    '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/policies/collect_human_27_09_rl_model_best',
    # '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/policies/collect_best_fm_23_08_rl_model_best'
    map_location=torch.device(device))
fm_network.load_state_dict(cleanppofm_model["_fm"])
# fm_network.load_state_dict(torch.load(
#     '/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/src/best_model_1',
#     map_location=torch.device(device)))
fm_network.eval()

predicted_right_positions_action_0 = 0
predicted_right_positions_action_0_without_zero_objects = 0
predicted_right_positions_action_1 = 0
predicted_right_positions_action_1_without_zero_objects = 0
predicted_right_positions_action_2 = 0
predicted_right_positions_action_2_without_zero_objects = 0
overall_number_of_positions_action_0_without_zero_objects = 0
overall_number_of_positions_action_1_without_zero_objects = 0
overall_number_of_positions_action_2_without_zero_objects = 0
overall_number_of_positions = torch.prod(torch.tensor(observations.shape)).item()

for index, element in enumerate(observations):
    active_belief_state_normal_distribution_action_0 = fm_network(element.unsqueeze(0),
                                                                  torch.tensor([[0]]).float())
    active_belief_state_normal_distribution_action_1 = fm_network(element.unsqueeze(0),
                                                                  torch.tensor([[1]]).float())
    active_belief_state_normal_distribution_action_2 = fm_network(element.unsqueeze(0),
                                                                  torch.tensor([[2]]).float())
    gold_label = get_next_position_observation_moonlander(observations=element.repeat(3, 1),
                                                          actions=torch.tensor([0, 1, 2]),
                                                          observation_width=40,
                                                          observation_height=30, agent_size=2,
                                                          maximum_number_of_objects=maximum_number_of_objects)
    
    for j, object_position in enumerate(gold_label[0]):
        if object_position != 0:
            overall_number_of_positions_action_0_without_zero_objects += 1
            if object_position == torch.round(active_belief_state_normal_distribution_action_0.mean[0, j]):
                predicted_right_positions_action_0_without_zero_objects += 1
    for j, object_position in enumerate(gold_label[1]):
        if object_position != 0:
            overall_number_of_positions_action_1_without_zero_objects += 1
            if object_position == torch.round(active_belief_state_normal_distribution_action_1.mean[0, j]):
                predicted_right_positions_action_1_without_zero_objects += 1
    for j, object_position in enumerate(gold_label[2]):
        if object_position != 0:
            overall_number_of_positions_action_2_without_zero_objects += 1
            if object_position == torch.round(active_belief_state_normal_distribution_action_2.mean[0, j]):
                predicted_right_positions_action_2_without_zero_objects += 1
    
    print(
        f"action 0: "
        f"predicted {gold_label[0].unsqueeze(0) == torch.round(active_belief_state_normal_distribution_action_0.mean[:, :-1])} "
        f"because the actual positions is {gold_label[0]} "
        f"and the predicted positions is {torch.round(active_belief_state_normal_distribution_action_0.mean[:, :-1])}")
    print(
        f"action 1: "
        f"predicted {gold_label[1] == torch.round(active_belief_state_normal_distribution_action_1.mean[:, :-1])} "
        f"because the actual positions is {gold_label[1]} "
        f"and the predicted positions is {torch.round(active_belief_state_normal_distribution_action_1.mean[:, :-1])}")
    print(
        f"action 2: "
        f"predicted {gold_label[2] == torch.round(active_belief_state_normal_distribution_action_2.mean[:, :-1])} "
        f"because the actual positions is {gold_label[2]} "
        f"and the predicted positions is {torch.round(active_belief_state_normal_distribution_action_2.mean[:, :-1])}")
    
    predicted_right_positions_action_0 += torch.count_nonzero(
        gold_label[0].unsqueeze(0) == torch.round(active_belief_state_normal_distribution_action_0.mean[:, :-1])).item()
    predicted_right_positions_action_1 += torch.count_nonzero(
        gold_label[1].unsqueeze(0) == torch.round(active_belief_state_normal_distribution_action_1.mean[:, :-1])).item()
    predicted_right_positions_action_2 += torch.count_nonzero(
        gold_label[2].unsqueeze(0) == torch.round(active_belief_state_normal_distribution_action_2.mean[:, :-1])).item()

print(
    f"Action 0: having predicted right {predicted_right_positions_action_0} out of {overall_number_of_positions} "
    f"resolves in an accuracy of {predicted_right_positions_action_0 / overall_number_of_positions}")
print(
    f"Action 0 - without zero objects: having predicted right {predicted_right_positions_action_0_without_zero_objects} "
    f"out of {overall_number_of_positions_action_0_without_zero_objects} "
    f"resolves in an accuracy of "
    f"{predicted_right_positions_action_0_without_zero_objects / overall_number_of_positions_action_0_without_zero_objects}")
print(
    f"Action 1: having predicted right {predicted_right_positions_action_1} out of {overall_number_of_positions} "
    f"resolves in an accuracy of {predicted_right_positions_action_1 / overall_number_of_positions}")
print(
    f"Action 1 - without zero objects: having predicted right {predicted_right_positions_action_1_without_zero_objects} "
    f"out of {overall_number_of_positions_action_1_without_zero_objects} "
    f"resolves in an accuracy of "
    f"{predicted_right_positions_action_1_without_zero_objects / overall_number_of_positions_action_1_without_zero_objects}")
print(
    f"Action 2: having predicted right {predicted_right_positions_action_2} out of {overall_number_of_positions} "
    f"resolves in an accuracy of {predicted_right_positions_action_2 / overall_number_of_positions}")
print(
    f"Action 2 - without zero objects: having predicted right {predicted_right_positions_action_2_without_zero_objects} "
    f"out of {overall_number_of_positions_action_2_without_zero_objects} "
    f"resolves in an accuracy of "
    f"{predicted_right_positions_action_2_without_zero_objects / overall_number_of_positions_action_2_without_zero_objects}")
