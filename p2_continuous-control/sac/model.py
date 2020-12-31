import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_layers):
        super(ValueNetwork, self).__init__()

        layers = []
        in_dim = num_inputs
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.layers = nn.ModuleList(layers)

        self.apply(weights_init_)

    def forward(self, state):
        x = state
        N = len(self.layers)

        for idx, linear in enumerate(self.layers):
            x = linear(x)
            if idx < N - 1:
                x = F.relu(x)

        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_layers):
        super(QNetwork, self).__init__()

        # Q1 architecture
        layers = []
        in_dim = num_inputs + num_actions
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.q1_layers = nn.ModuleList(layers)

        # Q2 architecture
        layers = []
        in_dim = num_inputs + num_actions
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.q2_layers = nn.ModuleList(layers)

        self.apply(weights_init_)

    def forward(self, state, action):
        x1 = x2 = torch.cat([state, action], 1)
        N = len(self.q1_layers)

        for idx, linear in enumerate(self.q1_layers):
            x1 = linear(x1)
            if idx < N - 1:
                x1 = F.relu(x1)

        for idx, linear in enumerate(self.q2_layers):
            x2 = linear(x2)
            if idx < N - 1:
                x2 = F.relu(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_layers, action_space=None):
        super(GaussianPolicy, self).__init__()

        layers = []
        in_dim = num_inputs
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)

        self.mean_linear = nn.Linear(hidden_layers[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_layers[-1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

    def forward(self, state):
        x = state
        for linear in self.layers:
            x = F.relu(linear(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        shifted_squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, shifted_squashed_mean, mean, std

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_layers, action_space=None):
        super(DeterministicPolicy, self).__init__()

        layers = []
        in_dim = num_inputs
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)

        self.mean = nn.Linear(hidden_layers[-1], num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

    def forward(self, state):
        x = state
        for linear in self.layers:
            x = F.relu(linear(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state, std_value=0.1):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=std_value)
        noise = noise.clamp(-0.25, 0.25)  # practically limit to +/- 2.5*stdev
        action = mean + noise
        return action, torch.tensor(0.0), mean, mean, torch.tensor([std_value] * state.shape[1])

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
