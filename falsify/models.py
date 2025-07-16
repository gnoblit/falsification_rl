import torch
import torch.nn as nn
from gymnasium.spaces import Discrete

class MiniGridCNN(nn.Module):
    """Shared feature extractor for MiniGrid."""
    def __init__(self, obs_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, kernel_size=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            self.feature_size = self.conv(dummy_input).shape[1]

    def forward(self, obs):
        return self.conv(obs)

class PolicyValueNet(nn.Module):
    """Actor-Critic network that takes features as input."""
    def __init__(self, feature_size: int, action_space: Discrete):
        super().__init__()
        self.critic = nn.Linear(feature_size, 1)
        self.actor = nn.Linear(feature_size, action_space.n)

    def forward(self, features):
        return self.actor(features), self.critic(features)

class TheoryModel(nn.Module):
    """M_Theory: Predicts the next state's features."""
    def __init__(self, feature_size: int, action_space: Discrete, hidden_size: int = 256):
        super().__init__()
        self.action_dim = action_space.n
        self.fc = nn.Sequential(
            nn.Linear(feature_size + self.action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        )

    def forward(self, state_features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Squeeze action tensor if it has an extra dimension
        if action.dim() > 1 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=self.action_dim).float()
        x = torch.cat([state_features, action_one_hot], dim=-1)
        return self.fc(x)

class FalsifierModel(nn.Module):
    """M_Falsifier: Predicts future Theory Model error from a planned trajectory."""
    def __init__(self, feature_size: int, action_space: Discrete, horizon: int, hidden_size: int = 256):
        super().__init__()
        self.action_dim = action_space.n
        self.gru = nn.GRU(
            input_size=feature_size + self.action_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, state_features_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        # Squeeze action tensor if it has an extra dimension
        if action_seq.dim() > 2 and action_seq.shape[-1] == 1:
            action_seq = action_seq.squeeze(-1)
        action_one_hot = torch.nn.functional.one_hot(action_seq, num_classes=self.action_dim).float()
        input_seq = torch.cat([state_features_seq, action_one_hot], dim=-1)
        gru_out, _ = self.gru(input_seq)
        last_hidden_state = gru_out[:, -1, :]
        return self.fc(last_hidden_state).squeeze(-1)