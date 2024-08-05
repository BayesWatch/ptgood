import gym
from dm_control import suite
import dmc2gym
import torch
import numpy as np
from networks.ensembles import DynamicsEnsemble
from utils.termination_fns import termination_fns
from utils.replays import ReplayBuffer, OfflineReplay
from rl.sac import SAC
import gym
import d4rl
import alternate_envs
from ml.mixtures import GMM
from ml.ceb import CEB
from utils.data import preprocess_sac_batch, preprocess_sac_batch_oto
from planners.ceb_planners import CEBVecTree
import argparse
from tqdm import tqdm
import json
import wandb
from copy import deepcopy


args = argparse.ArgumentParser()
args.add_argument('--env')
args.add_argument('--a_repeat', type=int, default=1)
args.add_argument('--custom_filepath', default=None)
args.add_argument('--horizon', type=int, default=1)
args.add_argument('--offline_steps', type=int, required=True)
args.add_argument('--online_steps', type=int, required=True)
args.add_argument('--model_file', default=None)
args.add_argument('--rl_file', default=None)
args.add_argument('--ceb_file', default=None)
args.add_argument('--ceb_beta', type=float)
args.add_argument('--ceb_z_dim', type=int)
args.add_argument('--ceb_planner', action='store_true')
args.add_argument('--ceb_planner_noise', type=float, default=0.0)
args.add_argument('--ceb_width', type=int)
args.add_argument('--ceb_depth', type=int)
args.add_argument('--ceb_update_freq', type=int, default=999999)
args.add_argument('--learned_marginal', action='store_true')
args.add_argument('--act_ceb_pct', type=float)
args.add_argument('--wandb_key')
args.add_argument('--reward_penalty', default=None)
args.add_argument('--reward_penalty_weight', default=1, type=float)
args.add_argument('--loss_penalty', default=None)
args.add_argument('--threshold', default=None, type=float)
args.add_argument('--model_notes', default=None, type=str)
args.add_argument('--eval_model', action='store_true')
args.add_argument('--r', default=0.5, type=float)
args.add_argument('--imagination_freq', type=int)
args.add_argument('--model_train_freq', type=int)
args.add_argument('--rollout_batch_size', type=int)
args.add_argument('--save_rl_post_online', action='store_true')
args.add_argument('--save_rl_post_offline', action='store_true')
args.add_argument('--rl_updates_per', type=int)
args.add_argument('--rl_grad_clip', type=float, default=999999999)
args.add_argument('--disagreement_weight', type=float, default=1.0)
args.add_argument('--critic_norm', action='store_true')
args.add_argument('--exp_name', type=str, default='oto-mbpo')
args.add_argument('--rl_initial_alpha', default=0.1, type=float)

if args.custom_filepath == 'None':
    args.custom_filepath = None

if not 'dmc2gym' in args.env:
    env = gym.make(args.env)
    eval_env = gym.make(args.env)

else:
    if 'walker' in args.env:
        env = dmc2gym.make(domain_name='walker', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)
        eval_env = dmc2gym.make(domain_name='walker', task_name='walk', from_pixels=False, frame_skip=args.a_repeat)

    elif 'hopper' in args.env:
        env = dmc2gym.make(domain_name='hopper', task_name='hop', from_pixels=False, frame_skip=args.a_repeat)
        eval_env = dmc2gym.make(domain_name='hopper', task_name='hop', from_pixels=False, frame_skip=args.a_repeat)

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

"""Replays"""
online_replay_buffer = ReplayBuffer(100000, state_dim, action_dim, device)

model_retain_epochs = 1
# rollout_batch_size = 512
epoch_length = args.model_train_freq
# model_train_freq = 250
rollout_horizon_schedule_min_length = args.horizon
rollout_horizon_schedule_max_length = args.horizon

base_model_buffer_size = int(model_retain_epochs
                            * args.rollout_batch_size
                            * epoch_length / args.model_train_freq)
max_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_max_length
min_model_buffer_size = base_model_buffer_size * rollout_horizon_schedule_min_length

# Initializing space for the MAX and then setting the pointer ceiling at the MIN
# Doing so lets us scale UP the model horizon rollout length during training
model_replay_buffer = ReplayBuffer(max_model_buffer_size, state_dim, action_dim, device)
model_replay_buffer.max_size = min_model_buffer_size # * 50

print(f'Model replay buffer capacity: {max_model_buffer_size}\n')

"""Model"""
termination_fn = termination_fns[args.env.split('-')[0]]
print(f'Using termination function: {termination_fn}')

if 'humanoid' in args.env.lower() or 'pen' in args.env or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env or 'kitchen' in args.env:
    bs = 1024
    if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
        dynamics_hidden = 200
    else:
        dynamics_hidden = 400
else:
    bs = 256
    dynamics_hidden = 200

print(f'Dynamics hidden: {dynamics_hidden}\n')

dynamics_ens = DynamicsEnsemble(
    7, state_dim, action_dim, [dynamics_hidden for _ in range(4)], 'elu', False, 'normal', 5000,
    True, True, 512, 0.001, 10, 5, None, False, args.reward_penalty, args.reward_penalty_weight, None, None, None,
    args.threshold, None, device
)

"""RL"""
if 'humanoid' in args.env.lower() or 'ant' in args.env.lower() or 'pen' in args.env or 'hammer' in args.env or 'door' in args.env or 'relocate' in args.env or 'quadruped' in args.env:
    if 'cmu' in args.env.lower() or 'escape' in args.env.lower():
        agent_mlp = [1024, 1024, 1024]
    else:
        agent_mlp = [512, 512, 512]
else:
    agent_mlp = [256, 256, 256]

print(f'Agent mlp: {agent_mlp}\n')
agent = SAC(
        state_dim, action_dim, agent_mlp, 'elu', args.critic_norm, -20, 2, 1e-4, 3e-4,
        3e-4, args.rl_initial_alpha, 0.99, 0.005, [-1, 1], 256, 2,
        2, None, device, args.rl_grad_clip
)

rl_batch_size = 512
real_ratio = args.r
n_eval_episodes = 20
online_ratio = args.r

"""CEB"""
if args.ceb_file:
    print(f'Loading CEB encoders from: {args.ceb_file}\n')

    try:
        ceb = CEB(state_dim, action_dim, [1024, 512, 512], args.ceb_z_dim, 'normal', args.ceb_beta, 'cuda')
        ceb.load(args.ceb_file)
        print(f'Large encoders: {[1024, 512, 512]}')
    except:
        ceb = CEB(state_dim, action_dim, [256, 128, 64], args.ceb_z_dim, 'normal', args.ceb_beta, 'cuda')
        ceb.load(args.ceb_file)
        print(f'Small encoders: {[256, 128, 64]}')

"""OFFLINE STUFF!"""
env.reset()
offline_replay = OfflineReplay(env, device, args.custom_filepath)

# NORMING! THIS DOES BOTH STATES AND ACTIONS...
print(f'SCALER B4: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

train_batch, _ = offline_replay.random_split(0, offline_replay.size)
train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
dynamics_ens.scaler.fit(train_inputs)

print(f'SCALER AFTER: {dynamics_ens.scaler.mean} / {dynamics_ens.scaler.std}')

dynamics_ens.replay = offline_replay

if args.ceb_planner:
    ceb.scaler = deepcopy(dynamics_ens.scaler)

"""LOGGING"""
with open(args.wandb_key, 'r') as f:
    API_KEY = json.load(f)['api_key']

import os
os.environ['WANDB_API_KEY'] = API_KEY
os.environ['WANDB_DATA_DIR'] = './wandb'
os.environ['WANDB_DIR'] = './wandb'
os.environ['WANDB_CONFIG_DIR'] = './wandb'
mode = 'online'

onfig = {'name': f'{args.env}_k{args.horizon}_m{args.model_notes}'}
wandb.init(
            project=args.exp_name,
            entity='{YOUR-ENTITY}',
            mode=mode,
            name=f'{args.env}_a{args.a_repeat}_k{args.horizon}_m{args.model_notes}_r{real_ratio}_online{args.online_steps}_{seed}',
            config=config
)

wandb.init()

"""PLANNING"""
if args.ceb_planner:
    """PLANNERS"""
    # planner = CEBTreePlanner()
    planner = CEBVecTree(lambda_q=0.0, lambda_r=1.0, noise_std=args.ceb_planner_noise)
    # planner = CEBParallelPlanner()
    planner.logger = wandb
    planner.termination_fn = termination_fn

dynamics_ens.logger = wandb
agent.logger = wandb

wandb.log({
    'ceb_planner_noise': args.ceb_planner_noise,
    'ceb_width': args.ceb_width,
    'ceb_depth': args.ceb_depth,
    'ceb_update_freq': args.ceb_update_freq
})


if args.model_file:
    print(f'Loading model file from {args.model_file}\n')
    dynamics_ens.load_state_dict(torch.load(args.model_file))

else:
    """Model Fitting"""
    print(f'No model file given. Training model from scratch...\n')

    model_fitting_steps = 0
    loss_ckpt = 999
    early_stop = 250
    early_stop_counter = 0

    while early_stop_counter < early_stop:
        loss_hist = dynamics_ens.train_single_step(dynamics_ens.replay, 0.2, bs)

        batch_size = 1024
        b_idx = 0
        e_idx = b_idx + batch_size
        state_error = []
        reward_error = []

        while e_idx <= dynamics_ens.replay.size:
            state = dynamics_ens.replay.states[b_idx: e_idx]
            action = dynamics_ens.replay.actions[b_idx: e_idx]
            next_state = dynamics_ens.replay.next_states[b_idx: e_idx]
            reward = dynamics_ens.replay.rewards[b_idx: e_idx]
            not_done = dynamics_ens.replay.not_dones[b_idx: e_idx]

            train_batch = (
                torch.FloatTensor(state).to('cuda'),
                torch.FloatTensor(action).to('cuda'),
                torch.FloatTensor(next_state).to('cuda'),
                torch.FloatTensor(reward).to('cuda'),
                torch.FloatTensor(not_done).to('cuda')
            )

            train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
            train_inputs = dynamics_ens.scaler.transform(train_inputs)

            with torch.no_grad():
                means, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                    train_inputs
                )

            state_err = (means - train_targets)[:, :-1].pow(2).mean().cpu().item()
            reward_err = (means - train_targets)[:, -1].pow(2).mean().cpu().item()
            state_error.append(state_err)
            reward_error.append(reward_err)

            b_idx += batch_size
            e_idx += batch_size

            if np.all([b_idx < dynamics_ens.replay.size, e_idx > dynamics_ens.replay.size]):
                e_idx = dynamics_ens.replay.size

        curr_loss = np.mean(state_error) + np.mean(reward_error)
        if loss_ckpt > curr_loss:
            loss_ckpt = curr_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # if model_fitting_steps < 1000:
        #     early_stop_counter = 0

        wandb.log({
            'model_early_stop': early_stop_counter,
            'model_loss': np.mean(loss_hist),
            'loss_ckpt': loss_ckpt,
            'curr_loss': curr_loss,
            'model_fitting_steps': model_fitting_steps,
            'state_err': np.mean(state_error),
            'reward_err': np.mean(reward_error)
        })
        model_fitting_steps += 1

        if model_fitting_steps % 1000 == 0:
            if not args.custom_filepath:
                extra = None
            elif 'random' in args.custom_filepath:
                extra = 'random'
            elif 'medium' in args.custom_filepath:
                extra = 'medium'
            elif 'medium-replay' in args.custom_filepath:
                extra = 'medium-replay'
            else:
                extra = None

            print(f'Saving model to: ./models/{args.env}_a{args.a_repeat}_{extra}_{seed}.pt\n')
            torch.save(
                dynamics_ens.state_dict(),
                f'./models/{args.env}_a{args.a_repeat}_{extra}_step{model_fitting_steps}_{seed}.pt'
            )

    # Done here
    if not args.custom_filepath:
        extra = None
    elif 'random' in args.custom_filepath:
        extra = 'random'
    elif 'medium' in args.custom_filepath:
        extra = 'medium'
    elif 'medium-replay' in args.custom_filepath:
        extra = 'medium-replay'
    else:
        extra = None

    print(f'Saving model to: ./models/{args.env}_a{args.a_repeat}_{extra}_{seed}.pt\n')
    torch.save(
        dynamics_ens.state_dict(),
        f'./models/{args.env}_a{args.a_repeat}_{extra}_step{model_fitting_steps}_{seed}.pt'
    )

"""Offline pre-training"""
eval_hist = []
offline_pretraining_step = 0

if not args.rl_file:
    print(f'No RL file given. Starting policy pre-training from offline dataset...\n')
    with tqdm(total=args.offline_steps) as pbar:
        while offline_pretraining_step <= args.offline_steps:
            # Eval policy
            eval_rewards = []
            if args.eval_model:
                model_errors_s = []
                model_errors_r = []

                for _ in range(n_eval_episodes):
                    eval_obs = env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action, dist = agent.act(eval_obs, sample=False, return_dist=True)
                        eval_next_obs, reward, done, info = env.step(action)
                        # wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})

                        model_input = torch.cat([
                            torch.from_numpy(eval_obs).float().to(dynamics_ens.device).unsqueeze(0),
                            torch.from_numpy(action).float().to(dynamics_ens.device).unsqueeze(0)
                        ], dim=-1)

                        model_input = dynamics_ens.scaler.transform(model_input)

                        with torch.no_grad():
                            model_pred = dynamics_ens.forward_models[
                                np.random.choice(dynamics_ens.selected_elites)
                            ](model_input, moments=False).sample()

                        next_state_pred = model_pred[:, :-1].cpu().numpy() + eval_obs
                        model_errors_s.append(((next_state_pred - eval_next_obs) ** 2).mean())
                        model_errors_r.append(((model_pred[:, -1].cpu().numpy() - reward) ** 2).mean())

                        episode_reward += reward
                        eval_obs = eval_next_obs

                    eval_rewards.append(episode_reward)

                wandb.log({
                    'model_error_s': np.mean(model_errors_s),
                    'model_error_r': np.mean(model_errors_r),
                    'step': offline_pretraining_step
                })

            else:
                for _ in range(n_eval_episodes):
                    eval_obs = env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action, dist = agent.act(eval_obs, sample=False, return_dist=True)
                        eval_next_obs, reward, done, info = env.step(action)
                        # wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})
                        # if np.any(['pen' in args.env, 'door' in args.env, 'hammer' in args.env, 'relocate' in args.env]):
                        if np.any(['pen' in args.env, 'hammer' in args.env, 'relocate' in args.env, 'door' in args.env]):
                            if info['goal_achieved']:
                                episode_reward = 1
                                done = True
                            else:
                                pass
                        else:
                            episode_reward += reward
                        eval_obs = eval_next_obs

                    eval_rewards.append(episode_reward)

            wandb.log(
                {'offline_step': offline_pretraining_step, 'offline_eval_returns': np.mean(eval_rewards)}
            )

            # Training loop
            for j in range(25000):
                # Need to start with filling the model_replay buffer a small amount
                dynamics_ens.imagine(
                    512,
                    args.horizon,
                    agent.actor,
                    offline_replay,
                    model_replay_buffer,
                    termination_fn,
                    offline_pretraining_step < 0
                )

                # The data used to update the policy is [(1-p)*imagined, p*real]
                agent.update(
                    preprocess_sac_batch(offline_replay, model_replay_buffer, rl_batch_size, real_ratio),
                    j,
                    args.loss_penalty,
                    None,
                    dynamics_ens
                )

                offline_pretraining_step += 1
                pbar.update(1)

    # Saving the RL model post-offline
    if args.save_rl_post_offline:
        print(f'Saving RL file to: ./policies/{args.env}_a{args.a_repeat}-k{args.horizon}_m{args.model_notes}_r{real_ratio}-{seed}-post_offline')
        agent.save(f'./policies/{args.env}_a{args.a_repeat}-k{args.horizon}_m{args.model_notes}_r{real_ratio}-{seed}-post_offline')

else:
    print(f'Loading RL file from {args.rl_file}\n')
    agent.load(args.rl_file)

    # Here in the logic flow, we have an empty model replay buffer. This is because we did not perform any offline
    # pre-training. So, let's just fill the model_replay_buffer with a handfull of trajectories using the frozen
    # policy from args.rl_file
    for _ in range(5):
        dynamics_ens.imagine(
            args.rollout_batch_size,
            args.horizon,
            agent.actor,
            offline_replay,
            model_replay_buffer,
            termination_fn,
            offline_pretraining_step < 0
        )

"""Online fine-tuning phase"""
online_steps = 0

# First, let's get a baseline of eval performance before any training
eval_rewards = []

for _ in range(n_eval_episodes):
    eval_obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, dist = agent.act(eval_obs, sample=False, return_dist=True)
        eval_next_obs, reward, done, info = env.step(action)
        # wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})

        # Measuring disagreement
        sa = torch.cat(
            [torch.FloatTensor(eval_obs).to(agent.device), torch.FloatTensor(action).to(agent.device)], dim=-1
        )
        oa = dynamics_ens.scaler.transform(sa)
        means = []
        for mem in dynamics_ens.selected_elites:
            mean, _ = dynamics_ens.forward_models[mem](
                oa
            )
            means.append(mean.unsqueeze(0))

        means = torch.cat(means, dim=0)
        disagreement = (torch.norm(means - means.mean(0), dim=-1)).mean(0).item()
        wandb.log({'disagreement': disagreement})

        if np.any(['pen' in args.env, 'door' in args.env, 'hammer' in args.env, 'relocate' in args.env]):
            if info['goal_achieved']:
                episode_reward = 1
                done = True
            else:
                pass
        else:
            episode_reward += reward

        eval_obs = eval_next_obs

    eval_rewards.append(episode_reward)

wandb.log(
    {'step': online_steps,
     'eval_returns': np.mean(eval_rewards)}
)

# (1) Filling the online_replay_buffer with an entire episode for seed steps. We carefully use random actions!
print(f'Prefilling online replay buffer with 1000 random steps...\n')
prefill = 0

while prefill < 1000:
    done = False
    obs = env.reset()

    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        online_replay_buffer.add(obs, action, reward, next_obs, done)
        prefill += 1
        obs = next_obs

        online_steps += 1

        if prefill >= 1000:
            break

if args.ceb_file:
    if args.learned_marginal:
        print(f'Fitting GMM marginal m(z)...')
        marginal = GMM(32, args.ceb_z_dim)
        marginal_opt = torch.optim.Adam(marginal.parameters(), lr=1e-3)

        for i in tqdm(range(50_000)):
            batch = model_replay_buffer.sample(512, True)
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

        ceb.marginal_z = marginal

    print(f'\nUpdating CEB global rate attribute...')
    # Want to update the global rate using samples from the policy's current occupancy measure,
    # which can be found in the model_replay_buffer
    ceb.update_global_rate(model_replay_buffer, scaler=ceb.scaler)


with tqdm(total=args.online_steps) as pbar:
    while online_steps <= args.online_steps:
        done = False
        obs = env.reset()
        episode_step = 0
        while not done:
            if args.ceb_planner:
                if np.random.rand() < args.act_ceb_pct:
                    action = planner.plan(
                        torch.FloatTensor(obs).unsqueeze(0).to(agent.device),
                        dynamics_ens, ceb, agent.actor, agent.critic, args.ceb_depth, args.ceb_width
                    )
                else:
                    action = agent.act(obs, sample=True)
            else:
                action = agent.act(obs, sample=True)

            next_obs, reward, done, info = env.step(action)
            episode_step += 1
            online_replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            for _ in range(args.rl_updates_per):
                agent.update(
                    preprocess_sac_batch_oto(
                        offline_replay, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio,
                        online_ratio
                    ),
                    online_steps,
                    args.loss_penalty,
                    [offline_replay, model_replay_buffer, online_replay_buffer, rl_batch_size, real_ratio,
                     online_ratio],
                    dynamics_ens
                )

            online_steps += 1
            pbar.update(1)

            if online_steps % args.model_train_freq == 0:
                train_batch, _ = offline_replay.random_split(0, offline_replay.size)
                online_batch, _ = online_replay_buffer.random_split(0, online_replay_buffer.size)
                train_batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in
                               zip(train_batch, online_batch)]

                train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
                dynamics_ens.scaler.fit(train_inputs)

                loss_ckpt = 999
                early_stop_ckpt = 5
                early_stop = 0

                while early_stop < early_stop_ckpt:
                    loss_hist = dynamics_ens.train_single_step(dynamics_ens.replay, 0.2, bs, online_replay_buffer)

                    batch_size = 1024
                    b_idx = 0
                    e_idx = b_idx + batch_size
                    state_error = []
                    reward_error = []

                    while e_idx <= dynamics_ens.replay.size:
                        state = dynamics_ens.replay.states[b_idx: e_idx]
                        action = dynamics_ens.replay.actions[b_idx: e_idx]
                        next_state = dynamics_ens.replay.next_states[b_idx: e_idx]
                        reward = dynamics_ens.replay.rewards[b_idx: e_idx]
                        not_done = dynamics_ens.replay.not_dones[b_idx: e_idx]

                        train_batch = (
                            torch.FloatTensor(state).to('cuda'),
                            torch.FloatTensor(action).to('cuda'),
                            torch.FloatTensor(next_state).to('cuda'),
                            torch.FloatTensor(reward).to('cuda'),
                            torch.FloatTensor(not_done).to('cuda')
                        )

                        train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
                        train_inputs = dynamics_ens.scaler.transform(train_inputs)

                        with torch.no_grad():
                            means, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                                train_inputs
                            )

                        state_err = (means - train_targets)[:, :-1].pow(2).mean().cpu().item()
                        reward_err = (means - train_targets)[:, -1].pow(2).mean().cpu().item()
                        state_error.append(state_err)
                        reward_error.append(reward_err)

                        b_idx += batch_size
                        e_idx += batch_size

                        if np.all([b_idx < dynamics_ens.replay.size, e_idx > dynamics_ens.replay.size]):
                            e_idx = dynamics_ens.replay.size

                    # Next over the online dataset. First need to reset indices
                    b_idx = 0
                    e_idx = b_idx + batch_size

                    while e_idx <= online_replay_buffer.size:
                        state = online_replay_buffer.states[b_idx: e_idx]
                        action = online_replay_buffer.actions[b_idx: e_idx]
                        next_state = online_replay_buffer.next_states[b_idx: e_idx]
                        reward = online_replay_buffer.rewards[b_idx: e_idx]
                        not_done = online_replay_buffer.not_dones[b_idx: e_idx]

                        train_batch = (
                            torch.FloatTensor(state).to('cuda'),
                            torch.FloatTensor(action).to('cuda'),
                            torch.FloatTensor(next_state).to('cuda'),
                            torch.FloatTensor(reward).to('cuda'),
                            torch.FloatTensor(not_done).to('cuda')
                        )

                        train_inputs, train_targets = dynamics_ens.preprocess_training_batch(train_batch)
                        train_inputs = dynamics_ens.scaler.transform(train_inputs)

                        with torch.no_grad():
                            means, _ = dynamics_ens.forward_models[np.random.choice(dynamics_ens.selected_elites)](
                                train_inputs
                            )

                        state_err = (means - train_targets)[:, :-1].pow(2).mean().cpu().item()
                        reward_err = (means - train_targets)[:, -1].pow(2).mean().cpu().item()
                        state_error.append(state_err)
                        reward_error.append(reward_err)

                        b_idx += batch_size
                        e_idx += batch_size

                        if np.all([b_idx < online_replay_buffer.size, e_idx > online_replay_buffer.size]):
                            e_idx = online_replay_buffer.size

                    # Now compare the error across our dataset to previous error checkpoints
                    curr_loss = np.mean(state_error) + np.mean(reward_error)
                    if loss_ckpt > curr_loss:
                        loss_ckpt = curr_loss
                        early_stop = 0
                    else:
                        early_stop += 1

                    wandb.log({
                        'model_early_stop': early_stop,
                        'model_loss': curr_loss,
                        'step': offline_pretraining_step + online_steps
                    })

            if online_steps % args.imagination_freq == 0:
                dynamics_ens.imagine(
                    args.rollout_batch_size,
                    args.horizon,
                    agent.actor,
                    online_replay_buffer,
                    model_replay_buffer,
                    termination_fn,
                    False
                )

            if online_steps % args.ceb_update_freq == 0:
                dynamics_ens.imagine(
                    args.rollout_batch_size,
                    args.horizon,
                    agent.actor,
                    online_replay_buffer,
                    model_replay_buffer,
                    termination_fn,
                    False
                )

                online_batch, _ = model_replay_buffer.random_split(0, model_replay_buffer.size)
                s, a, *_ = online_batch
                sa = torch.cat([s, a], dim=-1)

                ceb = CEB(state_dim, action_dim, [256, 128, 64], args.ceb_z_dim, 'normal', args.ceb_beta, 'cuda')
                ceb.scaler = deepcopy(dynamics_ens.scaler)
                ceb.scaler.fit(sa)

                for _ in range(50_000):
                    ceb_step_hist = ceb.train_step(512, model_replay_buffer, scaler=ceb.scaler)
                    wandb.log(ceb_step_hist)

                if args.learned_marginal:
                    marginal = GMM(32, args.ceb_z_dim)
                    marginal_opt = torch.optim.Adam(marginal.parameters(), lr=1e-3)

                    for i in range(30_000):
                        batch = model_replay_buffer.sample(512, True)
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

                    ceb.marginal_z = marginal

                ceb.update_global_rate(model_replay_buffer, scaler=ceb.scaler)

            if online_steps % 1000 == 0:
                eval_rewards = []
                for _ in range(n_eval_episodes):
                    eval_obs = eval_env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action, dist = agent.act(eval_obs, sample=False, return_dist=True)
                        eval_next_obs, reward, done, info = eval_env.step(action)
                        # wandb.log({'actor_entropy': dist.entropy().cpu().mean().item()})

                        # Measuring disagreement
                        sa = torch.cat(
                            [torch.FloatTensor(eval_obs).to(agent.device),
                             torch.FloatTensor(action).to(agent.device)], dim=-1
                        )
                        oa = dynamics_ens.scaler.transform(sa)
                        means = []
                        for mem in dynamics_ens.selected_elites:
                            mean, _ = dynamics_ens.forward_models[mem](
                                oa
                            )
                            means.append(mean.unsqueeze(0))

                        means = torch.cat(means, dim=0)
                        disagreement = (torch.norm(means - means.mean(0), dim=-1)).mean(0).item()
                        wandb.log({'disagreement': disagreement})

                        if np.any(['pen' in args.env, 'door' in args.env, 'hammer' in args.env,
                                   'relocate' in args.env]):
                            if info['goal_achieved']:
                                episode_reward = 1
                                done = True
                            else:
                                pass
                        else:
                            episode_reward += reward

                        eval_obs = eval_next_obs

                    eval_rewards.append(episode_reward)

                wandb.log(
                    {'step': online_steps,
                     'eval_returns': np.mean(eval_rewards)}
                )

            if args.save_rl_post_online:
                agent.save(
                    f'{args.env}_a{args.a_repeat}_bc{args.bc_policy}_k{args.horizon}_m{args.model_notes}_r{real_ratio}_online{args.online_steps}_{seed}')
