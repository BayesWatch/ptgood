import numpy as np
import torch
from planners.basic import Tree

class CEBVecTree:
    def __init__(self, lambda_q, lambda_r, noise_std):
        self.lambda_q = lambda_q
        self.lambda_r = lambda_r
        self.gamma = 0.99

        # Noise is used as a multiplier of torch.ones()
        self.noise_std = noise_std
        self.logger = None

    @torch.no_grad()
    def plan(self, state, dynamics_ens, ceb, policy, critic, depth, width):
        """

        Args:
            state:
            dynamics_ens:
            ceb:
            policy:
            critic:
            depth:
            width:

        Returns:

        """
        states = state.repeat(width, 1)
        actions = policy(states).sample()
        noise = torch.distributions.Normal(
            loc=torch.zeros_like(actions).to(actions.device),
            scale=torch.ones_like(actions).to(actions.device) * self.noise_std
        )
        actions += noise.sample()
        actions = torch.clamp(actions, -1., 1.)
        initial_actions = actions.clone()
        initial_action_rates = torch.zeros((1, width)).to(actions.device)
        initial_action_qs = torch.zeros((1, width)).to(actions.device)

        rate = ceb.compute_rate(
            ceb.scaler.transform(torch.cat([states, actions], dim=-1)),
            True
        )

        rate = (rate - ceb.rate_mean).abs()

        initial_action_rates += rate.reshape(1, width)

        for i in range(depth - 1):
            s_diff, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                dynamics_ens.scaler.transform(torch.cat([states, actions], dim=-1))
            )

            # Chopping the reward off of the s_diff and then adding it to the previous state to form s'
            states = states + s_diff[:, :-1]

            # Repeating states in this way allows us to .chunk(width) at the very end
            # Blocks are now [a1, a1, a1, ..., a2, a2, a2, ... a3, ...]
            # torch.cat([x.repeat(3, 1) for x in a.chunk(3, 0)], dim=0)
            states = torch.cat([x.repeat(width, 1) for x in states.chunk(width, 0)], dim=0)

            # Next actions
            actions = policy(states).sample()
            noise = torch.distributions.Normal(
                loc=torch.zeros_like(actions).to(actions.device),
                scale=torch.ones_like(actions).to(actions.device) * self.noise_std
            )
            actions += noise.sample()
            actions = torch.clamp(actions, -1., 1.)

            # bool True causes CEB encoder to return the mean of its parameterized dist
            rate = ceb.compute_rate(
                ceb.scaler.transform(torch.cat([states, actions], dim=-1)),
                True
            )

            # self.logger.log({'ceb_rate': ceb.rate_mean.item(), 'planning_rate': rate.mean().item()})

            # rate = (rate - ceb.rate_mean).abs() * self.gamma ** i
            rate = (rate - ceb.rate_mean) * self.gamma ** i

            # q1, q2 = critic(torch.cat([states, actions], dim=-1))
            # q_value = torch.min(q1, q2) * self.gamma ** i

            # Most term functions only use the final argument (next states)
            # We want to avoid purposefully selecting actions that terminate the episode
            if self.termination_fn:
                dones = self.termination_fn(states, actions, states)
                dones = dones.reshape(-1, 1)
                dones = 1 - dones.float()
                rate *= dones
                # q_value *= dones

            rate_per = torch.cat([x.mean().unsqueeze(0) for x in rate.chunk(width, 0)], dim=0)

            initial_action_rates += rate_per

        scores = initial_action_rates
        return initial_actions[scores.argmax()].cpu().numpy()
