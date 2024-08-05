import numpy as np
import torch

"""DONE BS: [bs, 1]"""


def hopper_term_fn(obs, act, next_obs):
    # assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = torch.isfinite(next_obs).all(-1) \
                * (next_obs[:, 1:].abs() < 100).all(-1) \
                * (height > .7) \
                * (torch.abs(angle) < .2)

    done = ~not_done
    done = done[:, None]
    return done


def walker2d_term_fn(obs, action, next_obs):
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) \
               * (height < 2.0) \
               * (angle > -1.0) \
               * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


def humanoidtruncobs_term_fn(obs, act, next_obs):
    # assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:, 0]
    done = torch.logical_or(z < 1.0, z > 2.0)
    # done = (z < 1.0) or (z > 2.0)

    # done = ~done
    done = done[:, None]
    return done


def fake_term_fn(obs, act, next_obs):
    # [100k, 1]
    return torch.ones((next_obs.shape[0], 1)).to(next_obs.device).bool()

def anttruncobs_term_fn(obs, act, next_obs):
    # not_terminated = (
    #         np.isfinite(next_obs).all() and next_obs[0] >= 0.2 and next_obs[0] <= 1.0
    # )
    first = torch.isfinite(next_obs).all(-1)
    second = next_obs[:, 0] > 0.2
    third = next_obs[:, 0] <= 1.0

    not_terminated = torch.logical_and(first, second).logical_and(third)

    done = not_terminated
    done = done[:, None]
    return done


termination_fns = {
    'halfcheetah': None,
    'walker2d': walker2d_term_fn,
    'hopper': hopper_term_fn,
    'walker2dnoterm': None,
    'hoppernoterm': None,
    'dmc2gym_walker': None,
    'dmc2gym_hopper': None,
    'dmc2gym_humanoid': None,
    'dmc2gym_quadruped': None,
    'pen': None,
    'hammer': None,
    'relocate': None,
    'door': None,
    'HumanoidTruncatedObs': humanoidtruncobs_term_fn,
    'HumanoidTruncatedObsMBPOReward': humanoidtruncobs_term_fn,
    'AntTruncatedObs': None
}
