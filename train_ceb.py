import torch
import torch.distributions as td
import alternate_envs
import numpy as np
import d4rl
import gym
from tqdm import tqdm
import argparse
import dmc2gym
import json
import wandb
from copy import deepcopy

from utils.replays import ReplayBuffer, OfflineReplay
from networks.ensembles import DynamicsEnsemble
from utils.termination_fns import termination_fns
from rl.sac import SAC
from ml.mixtures import GMM
from ml.ceb import CEB


args = argparse.ArgumentParser()
args.add_argument('--env')
args.add_argument('--a_repeat', type=int, default=1)
args.add_argument('--n_steps', type=int)
args.add_argument('--custom_filepath', default=None)
args.add_argument('--wandb_key')
args.add_argument('--bs', type=int)
args.add_argument('--traj_length', type=int)
args.add_argument('--large', action='store_true')
args.add_argument('--z_dim', type=int)
args.add_argument('--beta', type=float)
args.add_argument('--model_file', default=None)
args.add_argument('--rl_file', default=None)
args.add_argument('--horizon', type=int)
args.add_argument('--imagination_repeat', default=1, type=int)
args.add_argument('--rollout_batch_size', default=100000)
args.add_argument('--model_train_freq', default=250)
args.add_argument('--reward_penalty', default=None)
args.add_argument('--reward_penalty_weight', default=1, type=float)
args.add_argument('--threshold', default=None, type=float)
args.add_argument('--critic_norm', action='store_true')
args.add_argument('--rl_grad_clip', type=float, default=999999999)
args.add_argument('--ceb_file')
args.add_argument('--ceb_pretrained_file', type=str)
args = args.parse_args()

"""Environment"""
if not 'dmc2gym' in args.env:
    env = gym.make(args.env)
    dee4rl = True
    if 'pen' in args.env or 'ant' in args.env.lower() or 'humanoid':
        dee4rl = False

else:
    dee4rl = False

    if 'walker' in args.env:
        env = dmc2gym.make(domain_name='walker', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)

    elif 'hopper' in args.env:
        env = dmc2gym.make(domain_name='hopper', task_name='hop', from_pixels=False, frame_skip=args.a_repeat)

    elif 'humanoid' in args.env:
        if 'walk' in args.env:
            env = dmc2gym.make(domain_name='humanoid', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='humanoid', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)

        if 'run' in args.env:
            env = dmc2gym.make(domain_name='humanoid', task_name='run', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='humanoid', task_name='run', from_pixels=False, frame_skip=args.a_repeat)

        if 'stand' in args.env:
            env = dmc2gym.make(domain_name='humanoid', task_name='stand', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='humanoid', task_name='stand', from_pixels=False, frame_skip=args.a_repeat)

    elif 'quadruped' in args.env:
        if 'walk' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)

        elif 'run' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='run', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='run', from_pixels=False, frame_skip=args.a_repeat)

        elif 'escape' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='escape', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='escape', from_pixels=False, frame_skip=args.a_repeat)

        elif 'fetch' in args.env:
            env = dmc2gym.make(domain_name='quadruped', task_name='fetch', from_pixels=False, frame_skip=args.a_repeat)
            eval_env = dmc2gym.make(domain_name='quadruped', task_name='fetch', from_pixels=False, frame_skip=args.a_repeat)


seed = np.random.randint(0, 100000)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = 'cuda'

"""Offline replay"""
if dee4rl:
    if 'medium-replay' not in args.env:
        env = gym.make(args.env.replace('expert', 'random').replace('medium', 'random'))
        env.reset()
        offline_replay_rnd = OfflineReplay(env, device, args.custom_filepath)

        env = gym.make(args.env.replace('random', 'medium').replace('expert', 'medium'))
        env.reset()
        offline_replay_med = OfflineReplay(env, 'cuda')

        env = gym.make(args.env.replace('random', 'expert').replace('medium', 'expert'))
        env.reset()
        offline_replay_exp = OfflineReplay(env, 'cuda')

    else:
        env = gym.make(args.env.replace('medium-replay', 'random'))
        env.reset()
        offline_replay_rnd = OfflineReplay(env, device, args.custom_filepath)

        env = gym.make(args.env.replace('medium-replay', 'medium'))
        env.reset()
        offline_replay_med = OfflineReplay(env, 'cuda')

        env = gym.make(args.env.replace('medium-replay', 'expert'))
        env.reset()
        offline_replay_exp = OfflineReplay(env, 'cuda')


    if args.model_file and args.rl_file:
        model_retain_epochs = 1
        epoch_length = args.model_train_freq
        rollout_horizon_schedule_min_length = args.horizon
        rollout_horizon_schedule_max_length = args.horizon

        base_model_buffer_size = int(model_retain_epochs
                                     * args.rollout_batch_size
                                     * epoch_length / args.model_train_freq)
        max_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_max_length; min_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_min_length
        max_model_buffer_size *= args.imagination_repeat
        # Initializing space for the MAX and then setting the pointer ceiling at the MIN
        # Doing so lets us scale UP the model horizon rollout length during training
        training_replay = ReplayBuffer(max_model_buffer_size, state_dim, action_dim, device)
        training_replay.max_size = min_model_buffer_size  # * 50

        if 'humanoid' in args.env.lower() or 'pen' in args.env or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env:
            bs = 1024
            if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
                dynamics_hidden = 800
            else:
                dynamics_hidden = 400
        else:
            bs = 256
            dynamics_hidden = 200

        dynamics_ens = DynamicsEnsemble(
            7, state_dim, action_dim, [dynamics_hidden for _ in range(4)], 'elu', False, 'normal', 5000,
            True, True, 512, 0.001, 10, 5, None, False, args.reward_penalty, args.reward_penalty_weight, None, None,
            None,
            args.threshold, None, device
        )

        """RL"""
        # actor, critic, alpha
        # orig alpha init=0.1 [0.5]
        if 'humanoid' in args.env.lower() or 'ant' in args.env.lower() or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env:
            if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
                agent_mlp = [1024, 1024, 1024]
            else:
                agent_mlp = [512, 512, 512]
        else:
            agent_mlp = [256, 256, 256]

        print(f'Agent mlp: {agent_mlp}\n')

        agent = SAC(
            state_dim, action_dim, agent_mlp, 'elu', args.critic_norm, -20, 2, 1e-4, 3e-4,
            3e-4, 0.1, 0.99, 0.005, [-1, 1], 256, 2, 2, None, device, args.rl_grad_clip
        )

        print(f'Loading model file from {args.model_file}\n')
        dynamics_ens.load_state_dict(torch.load(args.model_file))

        if args.rl_file == 'rnd':
            print(f'Using random policy')
        else:
            print(f'Loading RL file from {args.rl_file}\n')
            agent.load(args.rl_file)

        termination_fn = termination_fns[args.env.split('-')[0]]
        print(f'Using termination function: {termination_fn}')

        print(f'WM capacity: {training_replay.capacity}')

        if 'random' in args.env:
            # NORMING! THIS DOES BOTH STATES AND ACTIONS...
            print(f'SCALER B4: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

            train_batch, _ = offline_replay_rnd.random_split(0, offline_replay_rnd.size)
            train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
            dynamics_ens.scaler.fit(train_inputs)

            print(f'SCALER AFTER: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

            dynamics_ens.replay = offline_replay_rnd


        elif 'medium' in args.env:
            # NORMING! THIS DOES BOTH STATES AND ACTIONS...
            print(f'SCALER B4: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

            train_batch, _ = offline_replay_med.random_split(0, offline_replay_med.size)
            train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
            dynamics_ens.scaler.fit(train_inputs)

            print(f'SCALER AFTER: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

            dynamics_ens.replay = offline_replay_med


        else:
            # NORMING! THIS DOES BOTH STATES AND ACTIONS...
            print(f'SCALER B4: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

            train_batch, _ = offline_replay_exp.random_split(0, offline_replay_exp.size)
            train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
            dynamics_ens.scaler.fit(train_inputs)

            print(f'SCALER AFTER: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

            dynamics_ens.replay = offline_replay_exp

            print(f'SCALER B4: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

            train_batch, _ = offline_replay_rnd.random_split(0, offline_replay_rnd.size)
            train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
            dynamics_ens.scaler.fit(train_inputs)

            print(f'SCALER AFTER: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

            dynamics_ens.replay = offline_replay_rnd

        # Performing rollout
        for _ in range(args.imagination_repeat):
            dynamics_ens.imagine(
                args.rollout_batch_size,
                args.horizon,
                agent.actor,
                dynamics_ens.replay,
                training_replay,
                termination_fn,
                False
            )

else:
    print(f'NOT IN DEE4RL!')
    env.reset()
    offline_replay_rnd = OfflineReplay(env, device, args.custom_filepath)

    if args.model_file and args.rl_file:
        model_retain_epochs = 1
        epoch_length = args.model_train_freq
        rollout_horizon_schedule_min_length = args.horizon
        rollout_horizon_schedule_max_length = args.horizon

        base_model_buffer_size = int(model_retain_epochs
                                     * args.rollout_batch_size
                                     * epoch_length / args.model_train_freq)
        max_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_max_length; min_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_min_length
        max_model_buffer_size *= args.imagination_repeat

        # Adding in the offline dataset
        max_model_buffer_size += offline_replay_rnd.actions.shape[0]
        # Initializing space for the MAX and then setting the pointer ceiling at the MIN
        # Doing so lets us scale UP the model horizon rollout length during training
        training_replay = ReplayBuffer(max_model_buffer_size, state_dim, action_dim, device)
        training_replay.max_size = min_model_buffer_size  # * 50

        if 'humanoid' in args.env.lower() or 'pen' in args.env or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env:
            bs = 1024
            if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
                dynamics_hidden = 800
            else:
                dynamics_hidden = 400
        else:
            bs = 256
            dynamics_hidden = 200

        dynamics_ens = DynamicsEnsemble(
            7, state_dim, action_dim, [dynamics_hidden for _ in range(4)], 'elu', False, 'normal', 5000,
            True, True, 512, 0.001, 10, 5, None, False, args.reward_penalty, args.reward_penalty_weight, None, None,
            None,
            args.threshold, None, device
        )

        """RL"""
        # actor, critic, alpha
        # orig alpha init=0.1 [0.5]
        if 'humanoid' in args.env.lower() or 'ant' in args.env.lower() or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env or 'pen' in args.env:
            if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
                agent_mlp = [1024, 1024, 1024]
            else:
                agent_mlp = [512, 512, 512]
        else:
            agent_mlp = [256, 256, 256]

        print(f'Agent mlp: {agent_mlp}\n')

        agent = SAC(
            state_dim, action_dim, agent_mlp, 'elu', args.critic_norm, -20, 2, 1e-4, 3e-4,
            3e-4, 0.1, 0.99, 0.005, [-1, 1], 256, 2, 2, None, device, args.rl_grad_clip
        )

        print(f'WM capacity: {training_replay.capacity}')

        print(f'Loading model file from {args.model_file}\n')
        dynamics_ens.load_state_dict(torch.load(args.model_file))

        if args.rl_file == 'rnd':
            print(f'Using random policy')
        else:
            print(f'Loading RL file from {args.rl_file}\n')
            agent.load(args.rl_file)

        termination_fn = termination_fns[args.env.split('-')[0]]
        print(f'Using termination function: {termination_fn}')

        print(f'SCALER B4: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

        train_batch, _ = offline_replay_rnd.random_split(0, offline_replay_rnd.size)
        train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
        dynamics_ens.scaler.fit(train_inputs)

        print(f'SCALER AFTER: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

        dynamics_ens.replay = offline_replay_rnd

        for _ in range(args.imagination_repeat):
            dynamics_ens.imagine(
                args.rollout_batch_size,
                args.horizon,
                agent.actor,
                dynamics_ens.replay,
                training_replay,
                termination_fn,
                False
            )

        training_replay.add_batch(obses=offline_replay_rnd.states, actions=offline_replay_rnd.actions,
                                  rewards=offline_replay_rnd.rewards, next_obses=offline_replay_rnd.next_states,
                                  not_dones=offline_replay_rnd.not_dones==0)


traj_length = args.traj_length
z_dim = args.z_dim

# [1024, 512, 512], [256, 128, 64]
if not args.large:
    print(f'S: {state_dim}')
    print(f'A: {action_dim}')
    ceb = CEB(state_dim, action_dim, [256, 128, 64], z_dim, 'normal', args.beta, 'cuda')

elif args.large:
    ceb = CEB(state_dim, action_dim, [1024, 512, 256, 128], z_dim, 'normal', args.beta, 'cuda')

# Adding wm scaler to CEB
ceb.scaler = deepcopy(dynamics_ens.scaler)


"""Logging"""
with open(args.wandb_key, 'r') as f:
    API_KEY = json.load(f)['api_key']

import os
os.environ['WANDB_API_KEY'] = API_KEY
os.environ['WANDB_DIR'] = './wandb'
os.environ['WANDB_CONFIG_DIR'] = './wandb'

wandb.init(
            project='ceb-training',
            # project='testing-wandb',
            entity='trevor-mcinroe',
            name=f'{args.env}_B{args.beta}_{seed}',
        )

if not args.ceb_pretrained_file:
    training_info = {'forward_i_xz_y': 0, 'log e(z_e|x)': 0, 'log b(z_e|y)': 0, 'forward_i_zy': 0,
                     'backward_i_xz_y': 0, 'log e(z_b|x)': 0, 'log b(z_b|y)': 0, 'backward_i_zx': 0,
                     'std_z': 0, 'mu_z': 0}
    tsc = 0
    for training_step in tqdm(range(args.n_steps)):

        # if training_step < 250_000:
        step_hist = ceb.train_step(args.bs, training_replay, scaler=ceb.scaler)
        tsc += 1
        for k in training_info:
            training_info[k] += step_hist[k]

        if training_step % 5000 == 0:
            for k, v in training_info.items():
                print(f'{k}: {v / tsc}')
            training_info = {'forward_i_xz_y': 0, 'log e(z_e|x)': 0, 'log b(z_e|y)': 0, 'forward_i_zy': 0,
                             'backward_i_xz_y': 0, 'log e(z_b|x)': 0, 'log b(z_b|y)': 0, 'backward_i_zx': 0,
                             'std_z': 0, 'mu_z': 0}
            tsc = 0
            if dee4rl:
                ceb.update_global_rate(offline_replay_rnd, scaler=ceb.scaler)
                step_hist['random_rate_std_upper'] = ceb.rate_mean + ceb.rate_std
                step_hist['random_rate_std_lower'] = ceb.rate_mean - ceb.rate_std

                ceb.update_global_rate(offline_replay_med, scaler=ceb.scaler)
                step_hist['medium_rate_std_upper'] = ceb.rate_mean + ceb.rate_std
                step_hist['medium_rate_std_lower'] = ceb.rate_mean - ceb.rate_std

                ceb.update_global_rate(offline_replay_exp, scaler=ceb.scaler)
                step_hist['expert_rate_std_upper'] = ceb.rate_mean + ceb.rate_std
                step_hist['expert_rate_std_lower'] = ceb.rate_mean - ceb.rate_std

        if dee4rl:
            # Medium
            batch = offline_replay_rnd.sample(args.bs, True)
            s, a, *_ = batch
            sa = torch.cat([s, a], dim=-1)
            sa = dynamics_ens.scaler.transform(sa)
            step_hist['random_rate'] = ceb.compute_rate(sa).mean().item()

            # Medium
            batch = offline_replay_med.sample(args.bs, True)
            s, a, *_ = batch
            sa = torch.cat([s, a], dim=-1)
            sa = dynamics_ens.scaler.transform(sa)
            step_hist['medium_rate'] = ceb.compute_rate(sa).mean().item()

            # Expert
            batch = offline_replay_exp.sample(args.bs, True)
            s, a, *_ = batch
            sa = torch.cat([s, a], dim=-1)
            sa = dynamics_ens.scaler.transform(sa)
            step_hist['expert_rate'] = ceb.compute_rate(sa).mean().item()

        wandb.log(step_hist)

else:
    print(f'Loading CEB weights from {args.ceb_pretrained_file}...\n')
    ceb.load(args.ceb_pretrained_file)
# Trying to train a marginal over the codespace
# Weights
from torch import nn
import torch.nn.functional as F

n_dists = 32

marginal = GMM(n_dists, args.z_dim)
marginal_opt = torch.optim.Adam(marginal.parameters(), lr=1e-3)

kl_loss = torch.nn.functional.kl_div

ceb.marginal_z = marginal

for i in tqdm(range(50_000)):
    batch = training_replay.sample(args.bs, True)
    s, a, *_ = batch
    sa = torch.cat([s, a], dim=-1)
    sa = ceb.scaler.transform(sa)

    z_dist = ceb.e_zx(sa, moments=False)
    z = z_dist.mean

    m_log_prob = marginal.log_prob(z).sum(-1, keepdim=True)
    loss = -m_log_prob.mean()

    marginal_opt.zero_grad()
    loss.backward()
    marginal_opt.step()

    wandb.log({'m_kl_loss': loss.item()})

    step_hist = {}
    batch = training_replay.sample(args.bs, True)
    s, a, *_ = batch
    sa = torch.cat([s, a], dim=-1)
    step_hist['rate'] = ceb.compute_rate(sa).mean().item()

    if dee4rl:
        batch = offline_replay_rnd.sample(args.bs, True)
        s, a, *_ = batch
        sa = torch.cat([s, a], dim=-1)
        step_hist['random_rate'] = ceb.compute_rate(sa).mean().item()

        # Medium
        batch = offline_replay_med.sample(args.bs, True)
        s, a, *_ = batch
        sa = torch.cat([s, a], dim=-1)
        step_hist['medium_rate'] = ceb.compute_rate(sa).mean().item()

        # Expert
        batch = offline_replay_exp.sample(args.bs, True)
        s, a, *_ = batch
        sa = torch.cat([s, a], dim=-1)
        step_hist['expert_rate'] = ceb.compute_rate(sa).mean().item()

    if i % 5000 == 0:
        if dee4rl:
            ceb.update_global_rate(training_replay, scaler=ceb.scaler)
            step_hist['training_rate_std_upper'] = ceb.rate_mean + ceb.rate_std
            step_hist['training_rate_std_lower'] = ceb.rate_mean - ceb.rate_std

            ceb.update_global_rate(offline_replay_rnd, scaler=ceb.scaler)
            step_hist['random_rate_std_upper'] = ceb.rate_mean + ceb.rate_std
            step_hist['random_rate_std_lower'] = ceb.rate_mean - ceb.rate_std

            ceb.update_global_rate(offline_replay_med, scaler=ceb.scaler)
            step_hist['medium_rate_std_upper'] = ceb.rate_mean + ceb.rate_std
            step_hist['medium_rate_std_lower'] = ceb.rate_mean - ceb.rate_std

            ceb.update_global_rate(offline_replay_exp, scaler=ceb.scaler)
            step_hist['expert_rate_std_upper'] = ceb.rate_mean + ceb.rate_std
            step_hist['expert_rate_std_lower'] = ceb.rate_mean - ceb.rate_std

    wandb.log(step_hist)

if args.ceb_file:
    ceb.save(args.ceb_file)
