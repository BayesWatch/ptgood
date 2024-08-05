import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as td
import math


class DistLayer(nn.Module):
    def __init__(self, input_shape, output_shape, dist):
        super().__init__()
        self.dist = dist
        self.lin_proj = nn.Linear(input_shape, output_shape)

        if dist in ['normal', 'trunc_normal']:
            self.std_proj = nn.Linear(input_shape, output_shape)

            self.min_logvar = nn.Parameter(
                torch.ones(output_shape) * -10,
                requires_grad=True
            )
            self.max_logvar = nn.Parameter(
                torch.ones(output_shape) * 0.5,
                requires_grad=True
            )

    def forward(self, x, moments):
        mu = self.lin_proj(x)

        if self.dist == 'normal':
            logvar = self.std_proj(x)
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

            if moments:
                return mu, logvar

            else:
                dist = td.Normal(mu, torch.sqrt(torch.exp(logvar)))
                return dist

        elif self.dist == 'mse':
            if moments:
                return mu, 1

            else:
                dist = td.Normal(mu, 1.0)
                return dist

        elif self.dist == 'normal_var_adjust':
            std = self.std_proj(x)
            std = 2 * torch.sigmoid((std + self.init_std) / 2)
            if moments:
                pass


        elif self.dist == 'trunc_normal':
            logvar = self.std_proj(x)
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

            if moments:
                return mu, logvar

            else:
                dist = SquashedNormal(mu, torch.sqrt(torch.exp(logvar)))
                return dist


class TanhTransform(td.transforms.Transform):
    domain = td.constraints.real
    codomain = td.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(td.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = td.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()
