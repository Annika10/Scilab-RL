import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from gymnasium import spaces
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProbabilisticSimpleForwardNet(nn.Module):
    def __init__(self, env, cfg):
        super().__init__()
        # FIXME:
        # hidden_size = cfg["hidden_size"]
        hidden_size = 256
        if isinstance(env.observation_space, spaces.dict.Dict):
            self.obs_shape = np.sum(
                [obs_space.shape for obs_space in env.observation_space.spaces.values()]
            )
        else:
            self.obs_shape = np.prod(env.observation_space.shape)
        self.action_shape = int(np.prod(env.action_space.shape))

        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.obs_shape + self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fw_mu = nn.Linear(hidden_size, self.obs_shape)
        self.fw_std = nn.Linear(hidden_size, self.obs_shape)

    def forward(self, obs, action):
        # forward model: p(w' | w, a)
        hx = torch.cat([obs, action], dim=-1)
        hx = self.state_action_encoder(hx)
        hx = F.relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())


class ProbabilisticSimpleForwardNetIncludingReward(nn.Module):
    def __init__(self, env, cfg):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        if isinstance(env.observation_space, spaces.dict.Dict):
            self.obs_shape = np.sum(
                [obs_space.shape for obs_space in env.observation_space.spaces.values()]
            )
        else:
            self.obs_shape = np.prod(env.observation_space.shape)
        self.action_shape = int(np.prod(env.action_space.shape))

        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.obs_shape + self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # obs_shape: int64
        # add one for reward
        self.fw_mu = nn.Linear(hidden_size, self.obs_shape + 1)
        self.fw_std = nn.Linear(hidden_size, self.obs_shape + 1)

    def forward(self, obs, action):
        # forward model: p(w' | w, a)
        hx = torch.cat([obs, action], dim=-1)
        hx = self.state_action_encoder(hx)
        hx = F.relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())


class ProbabilisticForwardNetPositionPredictionIncludingReward(nn.Module):
    def __init__(self, env, cfg, maximum_number_of_objects: int = 5):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        # 2 positions for the x, y coordinates of the agent
        # + 2 positions for the x, y coordinates of each object (maximum_number_of_objects)
        self.old_position_shape = 2 + 2 * maximum_number_of_objects
        self.action_shape = int(np.prod(env.action_space.shape))

        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.old_position_shape + self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # obs_shape: int64
        # add one for reward
        self.fw_mu = nn.Linear(hidden_size, self.old_position_shape + 1)
        self.fw_std = nn.Linear(hidden_size, self.old_position_shape + 1)

    def forward(self, old_position, action):
        # foward model: p(w' | w, a)
        hx = torch.cat([old_position.to(device), action.to(device)], dim=-1)
        hx = self.state_action_encoder(hx)
        hx = F.relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())


class ProbabilisticForwardNetPositionPrediction(nn.Module):
    def __init__(self, env, cfg, maximum_number_of_objects: int = 5):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        # 2 positions for the x, y coordinates of the agent
        # + 2 positions for the x, y coordinates of each object (maximum_number_of_objects)
        self.old_position_shape = 2 + 2 * maximum_number_of_objects
        self.action_shape = int(np.prod(env.action_space.shape))

        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.old_position_shape + self.action_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fw_mu = nn.Linear(hidden_size, self.old_position_shape)
        self.fw_std = nn.Linear(hidden_size, self.old_position_shape)

    def forward(self, old_position, action):
        # foward model: p(w' | w, a)
        hx = torch.cat([old_position, action], dim=-1)
        hx = self.state_action_encoder(hx)
        hx = F.relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            batch_norm: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=not batch_norm)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, bias=not batch_norm)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(in_channels)

        self._batch_norm = batch_norm

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.bn1(y) if self._batch_norm else y
        y = nn.functional.relu(y)
        y = self.conv2(y)
        y = self.bn2(y) if self._batch_norm else y
        return y + x


class SimpleConvModel(nn.Module):
    def __init__(
        self,
        env,
        cfg,
        **kwargs
    ):
        self.env = env
        self.config = cfg
        self.reward_function = kwargs.get("reward_function", None)

        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.input = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )

        self.residual_layer_1 = nn.Sequential(
            ResidualBlock(16 + 3, 32, batch_norm=True),
            nn.SiLU(),
            nn.Dropout(p=0.5)
        )

        self.residual_layer_2 = nn.Sequential(
            ResidualBlock(19 + 3, 32, batch_norm=True),
            nn.BatchNorm2d(22),
            nn.SiLU()
            #nn.Dropout(p=0.5)
        )

        if self.reward_function == "gaussian":
            self.reward_criterion = nn.MSELoss()
            self.reward_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(22 * 42 * 30, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
            )

        elif self.reward_function == "simple":
            self.reward_criterion = nn.CrossEntropyLoss()
            self.reward_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 42 * 30, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 3),
            )
        else:
            print("No reward prediction for this model")

        self.output = nn.Conv2d(22, 6, 1, 1, 0)

    def forward(self, x, act):
        # Convert to classes
        # Convert to onehot
        def form_observation_data_item_into_classes(data: np.array) -> Tensor:
            data_item_with_classes = np.select(
                [data == -10, data == -5, data == -1, data == 0, data == 1, np.isin(data, [2, 3])],
                [0, 1, 2, 3, 4, 5],
                default=-1
            )
            if np.any(data_item_with_classes == -1):
                raise ValueError(f"Invalid data_utils item: {np.argwhere(np.any(data_item_with_classes == -1))}")
            return torch.from_numpy(data_item_with_classes)

        # Convert observation to classes
        x = form_observation_data_item_into_classes(x)
        # Change shape (batch, width * height) -> (batch, width, height)
        x = x.reshape(-1, 30, 42).permute(0, 2, 1)
        # Onehot encode classes (batch, width, height) -> (batch, width, height, channels)
        x = F.one_hot(x, num_classes=6)
        # Reshape for conv input (batch, width, height, channels) -> (batch, channels, width, height)
        x = x.permute(0, 3, 1, 2)

        # Bring action to onehot format
        act = F.one_hot(act.long(), num_classes=3)
        # Repeat along spatial dimensions
        act = act.squeeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 42, 30)

        out = self.input(x.squeeze(1).float())
        out = self.residual_layer_1(torch.cat([out, act], dim=1))
        out = self.residual_layer_2(torch.cat([out, act], dim=1))
        reward_logits = None
        if self.reward_function is not None:
            reward_logits = self.reward_head(out)
        out = self.output(out)
        return out, reward_logits

    def loss_fn(self, *args, **kwargs):
        x_hat = args[0]
        x = args[1]
        reward_logits = args[2]
        reward_targets = args[3]
        state_loss = self.criterion(x_hat, x)
        loss = state_loss
        if self.reward_function is not None:
            reward_loss = self.reward_criterion(reward_logits, reward_targets.float()) * 0.001
            loss += reward_loss
        stats = {}
        return stats, loss


class ProbabilisticSimpleConv(nn.Module):
    def __init__(
        self,
        env,
        cfg,
        **kwargs
    ):
        super().__init__()
        if isinstance(env.observation_space, spaces.dict.Dict):
            self.obs_shape = np.sum(
                [obs_space.shape for obs_space in env.observation_space.spaces.values()]
            )
        else:
            self.obs_shape = np.prod(env.observation_space.shape)
        self.action_shape = int(np.prod(env.action_space.shape))

        # Output shape is (batch, 6, width, height)
        self.state_action_encoder = SimpleConvModel(env, cfg)

        # Change output shape through fully connected layers (batch, 6, width, height) -> (batch, 1 * width * height)
        self.fc = nn.Linear(6 * self.obs_shape, self.obs_shape)

        self.fw_mu = nn.Linear(self.obs_shape, self.obs_shape)
        self.fw_std = nn.Linear(self.obs_shape, self.obs_shape)

    def forward(self, obs, action):
        hx, hr = self.state_action_encoder(obs, action)
        hx = F.leaky_relu(hx)
        hx_flat = hx.reshape(obs.shape[0], -1)
        hx = self.fc(hx_flat)
        hx = F.leaky_relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return torch.distributions.Normal(fw_mu, fw_log_std.exp())
