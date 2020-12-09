import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=(64,64)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        layers = []
        num_inputs = state_size
        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(in_features=num_inputs, out_features=hidden_layer))
            layers.append(nn.ReLU())
            num_inputs = hidden_layer
        layers.append(nn.Linear(in_features=num_inputs, out_features=action_size))
        self.mlp = nn.Sequential(*layers)

        self.apply(weights_init_)

        # Initialize weights

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.mlp(state)
