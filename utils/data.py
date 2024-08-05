import torch
import numpy as np
import torch.distributions as td
from copy import deepcopy


def preprocess_sac_batch_oto(offline_buffer, model_buffer, online_buffer, batch_size, real_ratio, online_ratio):
    offline_bs = int(batch_size * real_ratio)
    model_bs = int(batch_size * (1 - real_ratio))
    online_bs = int(batch_size * online_ratio)

    offline_batch = offline_buffer.sample(offline_bs, rl=True)
    model_batch = model_buffer.sample(model_bs, rl=True)
    online_batch = online_buffer.sample(online_bs, rl=True)

    batch = [
        torch.cat((offline_item, model_item, online_item), dim=0)
        for offline_item, model_item, online_item in zip(offline_batch, model_batch, online_batch)
    ]

    return batch


def preprocess_sac_batch(env_replay_buffer, model_replay_buffer, batch_size, real_ratio):
    """"""
    env_batch_size = int(batch_size * real_ratio)
    model_batch_size = batch_size - env_batch_size

    env_batch = env_replay_buffer.sample(env_batch_size, rl=True)
    model_batch = model_replay_buffer.sample(model_batch_size, rl=True)

    batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
             zip(env_batch, model_batch)]
    return batch
