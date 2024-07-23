import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SimpleConvModel(nn.Module):
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
            nn.Dropout(p=0.5)
        )

        self.output = nn.Conv2d(22, 6, 1, 1, 0)

    def forward(self, x, act):
        act = act.squeeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 42, 30)
        out = self.input(x.float())
        out = self.residual_layer_1(torch.cat([out, act], dim=1))
        out = self.residual_layer_2(torch.cat([out, act], dim=1))
        out = self.output(out)
        return out

    def loss_fn(self, *args, **kwargs):
        x_hat = args[0]
        x = args[1]
        loss = self.criterion(x_hat, x)
        print(f"loss: {loss}")
        stats = {}
        return stats, loss


class ResidualBlock(nn.Module):
    """TODO docstring"""

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


class ProbabilisticSimpleForwardNet(nn.Module):
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
    def __init__(self, env, cfg):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        self.old_position_shape = 1
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
        hx = torch.cat([old_position, action], dim=-1)
        hx = self.state_action_encoder(hx)
        hx = F.relu(hx)
        fw_mu, fw_log_std = self.fw_mu(hx), self.fw_std(hx)
        fw_log_std = torch.clamp(fw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(fw_mu, fw_log_std.exp())


class ProbabilisticForwardNetPositionPrediction(nn.Module):
    def __init__(self, env, cfg):
        super().__init__()
        hidden_size = cfg["hidden_size"]
        self.old_position_shape = 1
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
