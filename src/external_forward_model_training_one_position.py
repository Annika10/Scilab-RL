import math
import torch
import gymnasium as gym
from src.custom_algorithms.cleanppofm.forward_model import ProbabilisticForwardNetOnePositionPrediction
from src.custom_envs.moonlander.utils import get_position_and_object_positions_of_observation, \
    get_next_position_observation_moonlander
from src.custom_envs.register_envs import register_custom_envs
import wandb


def train_fm(observations: torch.Tensor, actions: torch.Tensor, fm_network, fm_optimizer, maximum_number_of_objects,
             best_loss, criterion) -> tuple[float, float]:
    """
    Train the forward model with the actual next observations and rewards
    Args:
        observations: observations (Tensor: (64, 12/13))
        actions: action taken in observations leading to next_observations

    Returns:

    """
    observations = get_position_and_object_positions_of_observation(observations,
                                                                    maximum_number_of_objects=maximum_number_of_objects,
                                                                    observation_width=40, observation_height=30,
                                                                    agent_size=2)
    next_observations_formatted = get_next_position_observation_moonlander(observations=observations,
                                                                           actions=actions, observation_width=40,
                                                                           observation_height=30, agent_size=2,
                                                                           maximum_number_of_objects=maximum_number_of_objects)
    
    ##### FORWARD MODEL TRAINING #####
    # forward model prediction --> only agent
    forward_model_prediction = fm_network(observations[:, 0:1], actions.float().unsqueeze(1))
    # log probs is the logarithm of the maximum likelihood
    # log because the deviation is easier (addition instead of multiplication)
    # negative because likelihood normally maximizes
    fw_loss = -forward_model_prediction.log_prob(next_observations_formatted[:, 0:1])
    loss = fw_loss.mean()
    # loss = criterion(forward_model_prediction, next_object_position)
    # wandb.log({"loss": loss})
    
    # Track best performance, and save the model's state
    if loss < best_loss:
        print(f"New best loss: {loss.item()}")
        best_loss = loss
        model_path = 'best_model'
        torch.save(fm_network.state_dict(), model_path)
    
    fm_optimizer.zero_grad()
    loss.backward()
    fm_optimizer.step()
    return loss, best_loss


def train(fm_network, fm_optimizer, criterion, maximum_number_of_objects) -> None:
    """
    Update policy using the currently gathered rollout buffer.
    This implementation is mostly from the stable-baselines3 PPO implementation.
    """
    best_loss = math.inf
    
    # train for n_epochs epochs
    for epoch in range(2000):
        counter = 0
        tensor_of_observations = torch.tensor([])
        tensor_of_actions = torch.tensor([])
        
        for i in range(470):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                observation, info = env.reset()
            
            tensor_of_observations = torch.cat(
                (tensor_of_observations, torch.tensor(observation, device=device).unsqueeze(0)), dim=0)
            tensor_of_actions = torch.cat((tensor_of_actions, torch.tensor(action, device=device).unsqueeze(0)), dim=0)
        
        loss, best_loss = train_fm(observations=tensor_of_observations, actions=tensor_of_actions,
                                   fm_network=fm_network, fm_optimizer=fm_optimizer,
                                   maximum_number_of_objects=maximum_number_of_objects, best_loss=best_loss,
                                   criterion=criterion)
        print(f"EPOCH {epoch}, BATCH {counter} - LOSS: {loss.item()}")
        counter += 1


# run = wandb.init(
#     # Set the project where this run will be logged
#     project="external_forward_model_training",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.001,
#         "epochs": 20000,
#     },
# )

register_custom_envs()
fm_parameters = {'hidden_size': 256, 'learning_rate': 0.001, 'reward_eta': 0.2}
maximum_number_of_objects = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('MoonlanderWorld-dodge-gaussian-v0', reward_function="gaussian")
env.reset()

fm_network = ProbabilisticForwardNetOnePositionPrediction(env, fm_parameters).to(device)
fm_optimizer = torch.optim.Adam(
    fm_network.parameters(),
    lr=0.001
)
criterion = torch.nn.MSELoss()
# wandb.watch(fm_network, log_freq=100)
fm_network.train()

train(fm_network=fm_network, fm_optimizer=fm_optimizer, maximum_number_of_objects=maximum_number_of_objects,
      criterion=criterion)
