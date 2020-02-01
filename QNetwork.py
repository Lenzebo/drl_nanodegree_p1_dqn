import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.hidden_size = [200, 100]

        self.fc1 = nn.Linear(state_size, self.hidden_size[0])
        self.fc2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.fc3 = nn.Linear(self.hidden_size[1], action_size)

    def clone(self):
        clone = QNetwork(self.state_size, self.action_size, 123)
        return clone

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
