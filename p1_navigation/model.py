import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=(64,64)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list(int)): Number of nodes in each hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs = []
        in_size = state_size
        for layer_size in hidden_layers:
            self.fcs.append(nn.Linear(in_size, layer_size))
            in_size = layer_size
        self.fcs.append(nn.Linear(in_size, action_size))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for idx, fc in enumerate(self.fcs):
            if idx > 0:
                x = F.relu(x)
            x = fc(x)
        return x

    def save(self, ckpt_path="model.pt"):
        with open(ckpt_path, "wb") as fh:
            torch.save(self.state_dict(), fh)
        return ckpt_path

    def load(self, ckpt_path="model.pt"):
        with open(ckpt_path, "rb") as fh:
            self.load_state_dict(torch.load(fh))
