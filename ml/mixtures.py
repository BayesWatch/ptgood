import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as td


class GMM(nn.Module):
    def __init__(self, n_dists, z_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_dists).cuda(), requires_grad=True)
        self.mus = nn.Parameter(torch.randn((n_dists, z_dim)).cuda(), requires_grad=True)
        self.sigmas = nn.Parameter(torch.ones((n_dists, z_dim)).cuda(), requires_grad=True)

        self.min_logvar = nn.Parameter(
            torch.ones((n_dists, z_dim)).cuda() * -10,
            requires_grad=True
        )
        self.max_logvar = nn.Parameter(
            torch.ones((n_dists, z_dim)).cuda() * 0.5,
            requires_grad=True
        )

    def forward(self, x):
        logvar = self.max_logvar - F.softplus(self.max_logvar - self.sigmas)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        marginal = td.MixtureSameFamily(
            mixture_distribution=td.Categorical(self.weights.clip(0, 9999999)),
            component_distribution=td.Independent(td.Normal(loc=self.mus, scale=logvar), 1)
        )

        return marginal

    def log_prob(self, x):
        logvar = self.max_logvar - F.softplus(self.max_logvar - self.sigmas)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        marginal = td.MixtureSameFamily(
            mixture_distribution=td.Categorical(self.weights.clip(0, 9999999)),
            component_distribution=td.Independent(td.Normal(loc=self.mus, scale=logvar), 1)
        )

        return marginal.log_prob(x).unsqueeze(-1)
