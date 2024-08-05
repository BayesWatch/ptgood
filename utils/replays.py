import json

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.states = np.empty((capacity, obs_dim))
        self.next_states = np.empty((capacity, obs_dim))
        self.actions = np.empty((capacity, action_dim))
        self.rewards = np.empty((capacity, 1))
        self.not_dones = np.empty((capacity, 1))

        self.pointer = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, not_done):
        np.copyto(self.states[self.pointer], obs)
        np.copyto(self.actions[self.pointer], action)
        np.copyto(self.rewards[self.pointer], reward)
        np.copyto(self.next_states[self.pointer], next_obs)
        np.copyto(self.not_dones[self.pointer], 1 - not_done)

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obses, actions, rewards, next_obses, not_dones):
        for obs, action, reward, next_obs, not_done in zip(obses, actions, rewards, next_obses, not_dones):
            self.add(obs, action, reward, next_obs, not_done)

    def sample(self, batch_size, rl=False):
        if not rl:
            ind = np.random.choice(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)],
                size=batch_size
            )
        else:
            ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[ind]).to(self.device),
            torch.FloatTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.next_states[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.not_dones[ind]).to(self.device)
        )

    def sample_traj(self, batch_size, episode_length, horizon):
        eoo = np.where(self.not_dones[:self.size] == 0)[0]

        # Getting indexes where eoo != 0
        # [N, T] e.g., (N, 500) for when action_repeat = 2
        # traj_slices selects the beginning step in the subtrajectory
        traj_slices = np.random.choice(episode_length - horizon,
                                       size=len(eoo),
                                       replace=False)

        indexes = np.arange(self.size)

        indexes = np.array([
            indexes[eoo[i] + 1: eoo[i + 1]][traj_slices[i]: traj_slices[i] + horizon] if i > 0
            else indexes[eoo[i]: eoo[i + 1]][traj_slices[i]: traj_slices[i] + horizon]
            for i in range(len(eoo) - 1)
        ])

        batch_idxs = np.random.choice(indexes.shape[0], batch_size)
        batch_idxs = [indexes[batch_idxs]]

        training_batch = (
            torch.FloatTensor(self.states[batch_idxs]).to(self.device),
            torch.FloatTensor(self.actions[batch_idxs]).to(self.device),
            torch.FloatTensor(self.next_states[batch_idxs]).to(self.device),
            torch.FloatTensor(self.rewards[batch_idxs]).to(self.device),
            torch.FloatTensor(self.not_dones[batch_idxs]).to(self.device)
        )

        return training_batch

    def get_all(self):
        return (
            self.states[:self.pointer],
            self.actions[:self.pointer],
            self.next_states[:self.pointer],
            self.rewards[:self.pointer],
            self.not_dones[:self.pointer]
        )

    def random_split(self, val_size, batch_size=None):

        if batch_size is not None:
            batch_idxs = np.random.permutation(
                np.arange(self.size)[(self.not_dones[:self.size, :] != 2).reshape(-1)]
            )[:batch_size]

            training_batch = (
                torch.FloatTensor(self.states[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.actions[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.next_states[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.rewards[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.not_dones[batch_idxs[val_size:]]).to(self.device)
            )

            validation_batch = (
                torch.FloatTensor(self.states[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.actions[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.next_states[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.rewards[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.not_dones[batch_idxs[:val_size]]).to(self.device)
            )

        else:
            batch_idxs = np.random.permutation(
                np.arange(self.size)[(self.not_dones[:self.size, :] != 2).reshape(-1)]
            )

            training_batch = (
                torch.FloatTensor(self.states[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.actions[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.next_states[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.rewards[batch_idxs[val_size:]]).to(self.device),
                torch.FloatTensor(self.not_dones[batch_idxs[val_size:]]).to(self.device)
            )

            validation_batch = (
                torch.FloatTensor(self.states[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.actions[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.next_states[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.rewards[batch_idxs[:val_size]]).to(self.device),
                torch.FloatTensor(self.not_dones[batch_idxs[:val_size]]).to(self.device)
            )

        return training_batch, validation_batch

    @property
    def n_traj(self):
        return np.where(self.not_dones[:self.size] == 0)[0].shape[0]


class OfflineReplay:
    def __init__(self, env, device, custom_filepath=None):
        import d4rl
        self.env = env
        self.device = device

        if custom_filepath:
            with open(custom_filepath, 'r') as f:
                self.dataset = json.load(f)
                self.states = np.array(self.dataset['observations'])
                self.actions = np.array(self.dataset['actions'])
                self.next_states = np.array(self.dataset['next_observations'])
                self.rewards = np.array(self.dataset['rewards']).reshape(-1, 1)
                self.not_dones = (1 - np.array(self.dataset['terminals'])).reshape(-1, 1)

        else:
            self.dataset = d4rl.qlearning_dataset(env)
            self.states = self.dataset['observations']
            self.actions = self.dataset['actions']
            self.next_states = self.dataset['next_observations']
            self.rewards = self.dataset['rewards'].reshape(-1, 1)
            self.not_dones = (1 - self.dataset['terminals']).reshape(-1, 1)

        self.size = self.rewards.shape[0]

    def random_split(self, val_size, batch_size):
        # batch_idxs = np.random.permutation(self.size)[:batch_size]
        batch_idxs = np.random.permutation(
            np.arange(self.size)[(self.not_dones[:self.size, :] != 2).reshape(-1)]
        )[:batch_size]

        training_batch = (
            torch.FloatTensor(self.states[batch_idxs[val_size:]]).to(self.device),
            torch.FloatTensor(self.actions[batch_idxs[val_size:]]).to(self.device),
            torch.FloatTensor(self.next_states[batch_idxs[val_size:]]).to(self.device),
            torch.FloatTensor(self.rewards[batch_idxs[val_size:]]).to(self.device),
            torch.FloatTensor(self.not_dones[batch_idxs[val_size:]]).to(self.device)
        )

        validation_batch = (
            torch.FloatTensor(self.states[batch_idxs[:val_size]]).to(self.device),
            torch.FloatTensor(self.actions[batch_idxs[:val_size]]).to(self.device),
            torch.FloatTensor(self.next_states[batch_idxs[:val_size]]).to(self.device),
            torch.FloatTensor(self.rewards[batch_idxs[:val_size]]).to(self.device),
            torch.FloatTensor(self.not_dones[batch_idxs[:val_size]]).to(self.device)
        )

        return training_batch, validation_batch

    def sample(self, batch_size, rl=False):
        if not rl:
            ind = np.random.choice(
                np.arange(self.size)[(self.not_dones[:self.size, :] == 1).reshape(-1)],
                size=batch_size
            )
        else:
            ind = np.random.randint(0, self.size, size=batch_size)

        # print(f'INDS!: {ind}')
        # ind = np.random.randint(0, self.size, size=batch_size)[self.not_dones == 1]

        return (
            torch.FloatTensor(self.states[ind]).to(self.device),
            torch.FloatTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.next_states[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.not_dones[ind]).to(self.device)
        )
