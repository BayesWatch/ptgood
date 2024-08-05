import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pstar import pdict
import numpy as np

from torch.distributions import Uniform, Normal, Categorical
from networks.forward_models import MLP
from networks.distributions import DistLayer


class CEB(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, z_dim, dist, beta, device):
        super().__init__()

        self.e_zx = MLP(state_dim + action_dim, hidden_dims, z_dim, 'relu', False, dist)
        self.b_zy = MLP(state_dim + action_dim, hidden_dims, z_dim, 'relu', False, dist)

        self.marginal_z = Normal(loc=torch.tensor([0.]).to(device), scale=torch.tensor([1.]).to(device))

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.beta = beta

        self.rate_mean = 0
        self.rate_std = 0

        self.to(device)

        self.optim = torch.optim.Adam(
            list(self.e_zx.parameters()) + list(self.b_zy.parameters()),
            lr=1e-3
        )

    def train_step(self, bs, replay_buffer, scaler=None, second_replay_buffer=None):
        """
        Objective: beta * I(X;Z|Y) - I(Y;Z)
        Args:
            replay_buffer:

        Returns:

        """
        training_history = {'loss': None, 'i_xz_y': None, 'catgen': None, 'rate': None, 'std_z': None, 'mu_z': None}

        if second_replay_buffer:
            uniform_noise = Uniform(
                low=torch.ones([bs*2, self.state_dim + self.action_dim]).to(self.device) - 0.01,
                high=torch.ones([bs*2, self.state_dim + self.action_dim]).to(self.device) + 0.01
            )

        else:
            uniform_noise = Uniform(
                low=torch.ones([bs, self.state_dim + self.action_dim]).to(self.device) - 0.01,
                high=torch.ones([bs, self.state_dim + self.action_dim]).to(self.device) + 0.01
            )

        # This is really loud
        batch = replay_buffer.sample(bs, True)
        s, a, *_ = batch
        sa = torch.cat([s, a], dim=-1)

        if second_replay_buffer:
            batch = second_replay_buffer.sample(bs, True)
            s, a, *_ = batch
            sa2 = torch.cat([s, a], dim=-1)
            sa = torch.cat([sa, sa2], dim=0)

        if scaler:
            sa = scaler.transform(sa)

        second_sa = sa * uniform_noise.sample()

        # Use the reparameterization trick for gradients
        z_e_dist = self.e_zx(sa, moments=False)
        z_e = z_e_dist.rsample()

        log_z_e = z_e_dist.log_prob(z_e).sum(-1, keepdim=True)

        z_b_dist = self.b_zy(second_sa, moments=False)
        log_z_b = z_b_dist.log_prob(z_e).sum(-1, keepdim=True)

        i_xz_y = log_z_e - log_z_b
        training_history['forward_i_xz_y'] = i_xz_y.mean()
        training_history['log e(z_e|x)'] = log_z_e.mean()
        training_history['log b(z_e|y)'] = log_z_b.mean()
        training_history['std_z'] = z_e.std(0).mean()
        training_history['mu_z'] = z_e.mean(0).mean()

        def decoder_catgen_dist(decoder_dist_out, encoder_sample):
            dist = pdict()

            def log_prob():
                # print(f'encoder_sample: {encoder_sample.shape}')
                # print(f'encoder_sample[...]: {encoder_sample[..., None, :].shape}')
                logits = decoder_dist_out.log_prob(encoder_sample[..., None, :]).sum(-1, keepdim=True)
                logits = logits.squeeze(-1).unsqueeze(1)

                tau = 1.0
                if tau != 1.0:
                    logits = logits / tau

                dist.cat_dist = Categorical(logits=logits)

                indices = torch.arange(end=encoder_sample.shape[0]).reshape(-1, 1).cuda()
                log_probs = dist.cat_dist.log_prob(indices)
                mi_upper_bound = torch.log(torch.FloatTensor([encoder_sample.shape[-1]])).cuda()
                log_probs = log_probs + mi_upper_bound
                return log_probs

            dist.log_prob = log_prob
            return dist

        c_yz = decoder_catgen_dist(z_b_dist, z_e)
        i_zy = c_yz.log_prob()
        training_history['forward_i_zy'] = i_zy.mean()

        loss = self.beta * i_xz_y - i_zy
        loss = loss.mean()

        """Bidirectional"""
        z_b_dist = self.b_zy(second_sa, moments=False)
        z_b = z_b_dist.rsample()
        log_z_b = z_b_dist.log_prob(z_b).sum(-1, keepdim=True)

        z_e_dist = self.e_zx(sa, moments=False)
        log_z_e = z_e_dist.log_prob(z_b).sum(-1, keepdim=True)

        # i_xz_y = log_z_e - log_z_b
        i_xz_y = log_z_b - log_z_e
        training_history['backward_i_xz_y'] = i_xz_y.mean()
        training_history['log e(z_b|x)'] = log_z_e.mean()
        training_history['log b(z_b|y)'] = log_z_b.mean()

        def decoder_catgen_dist(decoder_dist_out, encoder_sample):
            dist = pdict()

            def log_prob():
                logits = decoder_dist_out.log_prob(encoder_sample[..., None, :]).sum(-1, keepdim=True)
                logits = logits.squeeze(-1).unsqueeze(1)

                tau = 1.0
                if tau != 1.0:
                    logits = logits / tau

                dist.cat_dist = Categorical(logits=logits)

                indices = torch.arange(end=encoder_sample.shape[0]).reshape(-1, 1).cuda()
                log_probs = dist.cat_dist.log_prob(indices)
                mi_upper_bound = torch.log(torch.FloatTensor([encoder_sample.shape[-1]])).cuda()
                log_probs = log_probs + mi_upper_bound
                return log_probs

            dist.log_prob = log_prob
            return dist

        # c_yz = decoder_catgen_dist(z_b_dist, z_e)
        c_yz = decoder_catgen_dist(z_e_dist, z_b)
        i_zy = c_yz.log_prob()
        training_history['backward_i_zx'] = i_zy.mean()

        loss2 = self.beta * i_xz_y - i_zy
        loss2 = loss2.mean()

        loss = loss + loss2

        training_history['loss'] = loss.item()
        training_history['i_xz_y'] = i_xz_y.mean().item()
        training_history['catgen'] = i_zy.mean().item()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        training_history['rate'] = self.compute_rate(sa).mean().item()
        return training_history

    def compute_rate(self, sa, detach=True):
        """
        Most previous works I have seen use an isotropic Gaussian for the variational approximation of p(z) -> m(z)
        R \equiv <log e(z|x)> - <log m(z)>

        Args:
            sa (torch.Tensor):

        Returns:

        """
        if detach:
            with torch.no_grad():
                z_e_dist = self.e_zx(sa, moments=False)
                # z_e = z_e_dist.sample()
                z_e = z_e_dist.mean
                log_z_e = z_e_dist.log_prob(z_e).sum(-1, keepdim=True)

                log_m_z = self.marginal_z.log_prob(z_e).sum(-1, keepdim=True)

        else:
            z_e_dist = self.e_zx(sa, moments=False)
            z_e = z_e_dist.rsample()
            log_z_e = z_e_dist.log_prob(z_e).sum(-1, keepdim=True)

            log_m_z = self.marginal_z.log_prob(z_e).sum(-1, keepdim=True)

        return log_z_e - log_m_z

    @torch.no_grad()
    def update_global_rate(self, replay, scaler=None, second_replay=None, return_all=False):
        """
        Loop over the entire replay and stores the mean rate
        Args:
            replay:

        Returns:

        """
        bs = 1024
        b_idx = 0
        e_idx = b_idx + bs
        rates = []

        while e_idx <= replay.size:
            state = torch.FloatTensor(replay.states[b_idx: e_idx]).to(self.device)
            action = torch.FloatTensor(replay.actions[b_idx: e_idx]).to(self.device)
            sa = torch.cat([state, action], dim=-1)

            if scaler:
                sa = scaler.transform(sa)

            rates.extend(self.compute_rate(sa, True).cpu().numpy().tolist())

            b_idx += bs
            e_idx += bs

            if np.all([b_idx < replay.size, e_idx > replay.size]):
                e_idx = replay.size

        if second_replay:
            bs = 1024
            b_idx = 0
            e_idx = b_idx + bs

            while e_idx <= second_replay.size:
                state = torch.FloatTensor(second_replay.states[b_idx: e_idx]).to(self.device)
                action = torch.FloatTensor(second_replay.actions[b_idx: e_idx]).to(self.device)
                sa = torch.cat([state, action], dim=-1)

                rates.extend(self.compute_rate(sa, True).cpu().numpy().tolist())

                b_idx += bs
                e_idx += bs

                if np.all([b_idx < second_replay.size, e_idx > second_replay.size]):
                    e_idx = second_replay.size

        self.rate_mean = np.mean(rates)
        self.rate_std = np.std(rates)

        if return_all:
            return rates

    def save(self, fname):
        torch.save(self.e_zx.state_dict(), f'{fname}_e_zx.pt')
        torch.save(self.b_zy.state_dict(), f'{fname}_b_zy.pt')

    def load(self, fname):
        self.e_zx.load_state_dict(torch.load(f'{fname}_e_zx.pt'))
        self.b_zy.load_state_dict(torch.load(f'{fname}_b_zy.pt'))
