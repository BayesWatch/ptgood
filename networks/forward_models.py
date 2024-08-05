import torch
from torch import nn
from networks.utils.activations import get_activation
from networks.distributions import DistLayer


class MLP(nn.Module):
    def __init__(self, input_shape, hidden_dims, output_shape, activation, norm, dist):
        """

        Args:
            input_shape:
            hidden_dims:
            output_shape:
            activation:
            norm:
            dist:
        """
        super().__init__()
        self.dist = dist
        self.layers = nn.ModuleList([nn.Linear(input_shape, hidden_dims[0])])

        for i in range(1, len(hidden_dims)):
            if norm:
                self.layers.append(nn.LayerNorm(hidden_dims[i - 1]))

            self.layers.append(get_activation(activation))
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        # TODO: if the DistLayer does not implement some linear projection + bias, this is probably not needed
        self.layers.append(get_activation(activation))

        if dist:
            self.dist_layer = DistLayer(hidden_dims[-1], output_shape, dist)

    @property
    def max_logvar(self):
        return self.dist_layer.max_logvar

    @property
    def min_logvar(self):
        return self.dist_layer.min_logvar

    def forward(self, x, moments=True):
        for layer in self.layers:
            x = layer(x)

        if self.dist:
            x = self.dist_layer(x, moments)

        return x


class GRU(nn.Module):
    def __init__(self, obs_dim, action_dim, reward_included, norm, activation, dist, update_bias=-1):
        super().__init__()
        self.dist = dist
        self._matrix = nn.Linear(obs_dim + action_dim, obs_dim * 3, bias=not norm)
        self._activation = get_activation(activation)
        self._obs_dim = obs_dim
        self._update_bias = update_bias
        self._reward_included = reward_included

        self._norm = norm
        if norm:
            self._norm = nn.LayerNorm(obs_dim * 3)

        if dist:
            self.dist_layer = DistLayer(obs_dim, obs_dim + reward_included, dist)

    def get_initial_state(self, batch_size):
        """"""
        return torch.zeros((batch_size, self._obs_dim))

    @property
    def max_logvar(self):
        return self.dist_layer.max_logvar

    @property
    def min_logvar(self):
        return self.dist_layer.min_logvar

    def forward(self, inputs, state, moments=True):
        h = self._matrix(torch.cat([inputs, state], dim=-1))

        if self._norm:
            h = self._norm(h)

        split = torch.split(h, [self._obs_dim, self._obs_dim, self._obs_dim], -1)
        reset, candidate, update = split[0], split[1], split[2]
        reset = torch.sigmoid(reset)
        candidate = self._activation(reset * candidate)
        update = torch.sigmoid(update + self._update_bias)

        output = update * candidate + (1 - update) * state

        if self.dist:
            output = self.dist_layer(output, moments)

        return output
