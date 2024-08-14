import numpy as np
import math
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions.categorical import Categorical
from stable_baselines3.common.logger import Logger
from custom_algorithms.cleanppofm.utils import flatten_obs, layer_init, get_position_of_observation
from custom_algorithms.cleanppofm.utils import form_observation_data_item_into_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):
    """
    Agent class for the PPO algorithm. The agent has a critic and an actor network. It only works with discrete actions.
    """

    def __init__(self, env) -> None:
        """
        Initialize the agent
        Args:
            env: using env for the observation space and action space size
        """
        super().__init__()
        # this is implemented for the gridworld envs
        if isinstance(env.observation_space, spaces.Dict):
            obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
            self.flatten = True
        # this is implemented for the moonlander env?
        else:
            obs_shape = np.array(env.observation_space.shape).prod()
            self.flatten = False

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, env.action_space.n))
        self.env = env

    def get_value(self, obs) -> torch.Tensor:
        """
        Get the value of the critic network
        Args:
            obs: observation

        Returns: value of the critic network in a tensor

        """
        if self.flatten:
            obs = flatten_obs(obs)
        else:
            obs = torch.tensor(obs, device=device, dtype=torch.float32).detach().clone()
        return self.critic(obs)

    def get_action_and_value_and_forward_model_prediction_and_prediction_error(self, fm_network, obs, env_name: str,
                                                                               action=None,
                                                                               deterministic: bool = False,
                                                                               logger: Logger = None,
                                                                               position_predicting: bool = False,
                                                                               use_new_forward_model: bool = True) -> \
            tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Normal, float]:
        """
        Get the action, logarithmic probability of action distribution, entropy of the action distribution
        value of the critic network and forward model prediction in form of a normal distribution
        Args:
            fm_network: instance of the forward model network
            obs: observation
            action: optional action to take
            deterministic: if true the action selection is deterministic and not stochastic
            logger: logger for logging
            position_predicting: if true the forward model predicts the position of the agent out of the observation

        Returns:
            action, logarithmic probability of action distribution, entropy of the action distribution, value of the
             critic network and forward model prediction in form of a normal distribution
        """

        ##### FLATTEN OBS #####
        # flatten the obs when it is a dict
        if self.flatten:
            obs = flatten_obs(obs)
        else:
            obs = torch.tensor(obs, device=device, dtype=torch.float32).clone().detach()

        ##### PREDICT NEXT ACTION #####
        # obs is a tensor
        # get the mean of each action (discrete) from the actor network
        # e.g. a tensor of size (1, 8) for each of the 8 discrete actions in the gridworld envs
        action_mean = self.actor_mean(obs)

        # create a categorical distribution from the mean of the actions
        # e.g. a tensor of size (1, 8) for the probability of each of 8 discrete actions in the gridworld envs
        # the probabilities equal to 1
        # From Wikipedia, the free encyclopedia:
        # In probability theory and statistics, a categorical distribution (also called a generalized Bernoulli
        # distribution, multinoulli distribution[1]) is a discrete probability distribution that describes the possible
        # results of a random variable that can take on one of K possible categories, with the probability of each
        # category separately specified. There is no innate underlying ordering of these outcomes, but numerical labels
        # are often attached for convenience in describing the distribution, (e.g. 1 to K). The K-dimensional
        # categorical distribution is the most general distribution over a K-way event; any other discrete distribution
        # over a size-K sample space is a special case. The parameters specifying the probabilities of each possible
        # outcome are constrained only by the fact that each must be in the range 0 to 1, and all must sum to 1.
        distribution = Categorical(logits=action_mean)

        # sample an action from the distribution if action is not none
        if action is None:
            if deterministic:
                action = torch.argmax(action_mean)
                forward_normal_action = action.unsqueeze(0).unsqueeze(0)
            else:
                action = distribution.sample()
                forward_normal_action = action.unsqueeze(0)
        else:
            forward_normal_action = action.unsqueeze(1)

        ##### PREDICT NEXT STATE #####
        # predict next state from last state and selected action
        # formal_normal_action in form of tensor([[action]])
        # get position of last state out of the observation --> moonlander specific implementation
        if use_new_forward_model:
            obs = np.transpose(obs.reshape(1, 30, 42), (0, 2, 1))
            obs = form_observation_data_item_into_classes(obs)
            obs = torch.nn.functional.one_hot(obs, num_classes=6).permute(0, 3, 1, 2)
            forward_model_prediction_normal_distribution = fm_network(obs, torch.nn.functional.one_hot(forward_normal_action[0], num_classes=3).float())
        else:
            if position_predicting:
                positions = get_position_of_observation(obs)
                forward_model_prediction_normal_distribution = fm_network(positions, forward_normal_action.float())
            else:
                forward_model_prediction_normal_distribution = fm_network(obs, forward_normal_action.float())

        ##### CALCULATE PREDICTION ERROR #####
        # prediction error version one -> standard deviation
        # prediction_error = forward_normal.stddev.mean().item()
        # prediction error version two -> Euclidean distance
        # calculate manually prediction error (Euclidean distance)
        if env_name == "GridWorldEnv":
            # predicted location can only be between 0 and 4
            max_distance_in_gridworld = math.sqrt(((4 - 0) ** 2) + ((4 - 0) ** 2) + ((4 - 0) ** 2) + ((4 - 0) ** 2))
            predicted_location = torch.tensor(
                [min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][0])), 4),
                 min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][1])), 4),
                 min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][2])), 4),
                 min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][3])), 4)])
            prediction_error = (math.sqrt(torch.sum((predicted_location - obs) ** 2))) / max_distance_in_gridworld
        elif env_name == "MoonlanderWorldEnv":
            # we just care for the x position of the moonlander agent, because the y position is always equally to the size of the agent
            if use_new_forward_model:
                # TODO: What to do here?
                pass
            else:
                if position_predicting:
                    positions = get_position_of_observation(obs)

                    # Smallest x position of the agent is the size of the agent
                    # Biggest x position of the agent is the width of the moonlander world - the size of the agent
                    first_possible_x_position = self.env.get_attr("first_possible_x_position")[0]
                    last_possible_x_position = self.env.get_attr("last_possible_x_position")[0]
                    max_distance_in_moonlander_world = math.sqrt(
                        (last_possible_x_position - first_possible_x_position) ** 2)
                    predicted_x_position = torch.tensor([min(max(first_possible_x_position,
                                                                 forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[
                                                                     0][0]),
                                                             last_possible_x_position)])
                    prediction_error = (math.sqrt(
                        torch.sum((predicted_x_position - positions[0]) ** 2))) / max_distance_in_moonlander_world
                # TODO: implement prediction error calculation for moonlander world env with predicting whole observation
                else:
                    raise NotImplementedError(
                        "Prediction error calculation not implemented for MoonlanderWorldEnv with predicting whole observation!")
        else:
            raise ValueError("Environment not supported")

        # TODO: put prediction of fm network into observation? --> standard deviation or whole observation?
        ##### LOGGING #####
        # std describes the (un-)certainty of the prediction of each pixel
        # more logging possible
        logger.record_mean("fm/stddev", forward_model_prediction_normal_distribution.stddev.mean().item())

        ##### RETURN #####
        # return predicted action, logarithmic probability of action distribution (one value in tensor),
        #  WIKIPEDIA Entropy: die durchschnittliche Anzahl von Entscheidungen (bits), die benötigt werden,
        #  um ein Zeichen aus einer Zeichenmenge zu identifizieren oder zu isolieren,
        #  anders gesagt: ein Maß, welches für eine Nachrichtenquelle den mittleren Informationsgehalt ausgegebener
        #  Nachrichten angibt.
        #  - sum of probabilities multiplied with the log_2 of the probabilities (one value in tensor)
        #  EXPLANATION ON REDDIT:
        #  It's a measure of how sure your classifier thinks it is.
        #  It maps the output vector between 0 (perfectly confident classification) and -ln(1/N),
        #  a large positive number (perfectly clueless classification).
        #  If your classifier puts all the weight onto one class, then it has minimal entropy
        #  (it believes it knows the exact class). If your classifier is totally unsure which class the example
        #  belongs to (every class' probability is equal), the entropy is maximal.
        #  There's not really a right answer to "what you want".
        #  You want the classifier to be right and sure in clear examples (so both correct and minimal entropy),
        #  but if you feed your cat vs dog classifier a picture containing both a cat and a dog,
        #  you probably want the model to give you a 50-50 split which is maximal entropy.
        # value of critic network, forward model prediction in normal distribution
        return action.unsqueeze(0), distribution.log_prob(action), distribution.entropy(), self.critic(
            obs), forward_model_prediction_normal_distribution, prediction_error
