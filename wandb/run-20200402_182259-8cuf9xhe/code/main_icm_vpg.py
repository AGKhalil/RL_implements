import numpy as np
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import time
import cProfile

import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

import wandb
import imageio

from icm import ICM
from vpg import VPG
from policies import MLP


def visualize():
    done = False
    obs = env.reset()
    imgs, visited_pos, visited_vel = [], [], []
    img = env.render('rgb_array')
    while not done:
        imgs.append(img)
        visited_pos.append(obs[0])
        visited_vel.append(obs[1])
        act, _ = vpg.get_action(obs)
        obs, rew, done, _ = env.step(act)
        img = env.render('rgb_array')

    imageio.mimsave('/tmp/current_gif.gif', [np.array(img) for i, img in enumerate(imgs) if i%2 == 0], fps=29)

    return visited_pos, visited_vel

def net_layers():
    if env_type == 'DISCRETE':
        act_space = env.action_space.n
    else:
        act_space = env.action_space.shape[0]
    obs_space = env.observation_space.shape[0]
    return [obs_space, 32, 16, act_space]


wandb.init(entity="agkhalil", project="pytorch-vpg-icm-mountaincarcont")
wandb.watch_called = False

config = wandb.config
config.episodes = 1000
config.lr = 0.0005
config.seed = 42
config.gamma = 0.99
eps = np.finfo(np.float32).eps.item()

device = torch.device('cpu')
torch.manual_seed(config.seed)
learning_rate = config.lr
batch_size = config.batch_size

env = gym.make('MountainCarContinuous-v0')
env_type = 'CONT'

mlp = MLP(net_layers()).to(device)
vpg = VPG(mlp, env, device, env_type)
icm = ICM(env, device)
optimizer_icm = optim.Adam(icm.parameters(), lr=learning_rate)
optimizer = optim.Adam(vpg.policy.parameters(), lr=learning_rate)
loss_inv_fun = torch.nn.MSELoss()
loss_for_fun = torch.nn.MSELoss()

EPISODES = config.episodes
gamma = config.gamma

wandb.watch(vpg.policy, log="all")

visited_pos, visited_vel = [], []

for episode in tqdm(range(0, EPISODES)):
    rewards = []
    ext_rewards = []
    int_rewards = []
    log_soft = []
    obs_lats = []
    next_obs_lats = []
    next_obs_lat_hats = []
    act_hats = []
    acts = []
    obs = env.reset()
    done = False
    in_reward = 0
    ex_reward = 0
    ep_reward = 0
    reward = 0
    while not done:
        action, log_prob = vpg.get_action(obs)
        new_obs, rew, done, _ = env.step(action)
        obs_lat, next_obs_lat, next_obs_lat_hat, act_hat = icm.forward(obs, new_obs)
        acts.append(action)
        act_hats.append(act_hat.squeeze(0))
        obs_lats.append(obs_lat)
        next_obs_lats.append(next_obs_lat)
        next_obs_lat_hats.append(next_obs_lat_hat)
        ext_rewards.append(rew)
        log_soft.append(log_prob)
        int_rewards.append(loss_for_fun(next_obs_lat, next_obs_lat_hat))
        in_reward += loss_for_fun(next_obs_lat, next_obs_lat_hat)
        ex_reward += rew
        ep_reward += rew + next_obs_lat - next_obs_lat_hat
        obs = new_obs

    rewards_size = len(ext_rewards)
    rewards = np.add(int_rewards, ext_rewards)
    gammas = [np.power(gamma, i) for i in range(rewards_size)]
    discounted_rewards = [np.sum(np.multiply(gammas[:rewards_size-i], rewards[i:])) for i in range(rewards_size)]
    optimizer.zero_grad()
    discounted_rewards = torch.tensor(discounted_rewards).to(device)
    advantage = discounted_rewards #- discounted_rewards.mean()) / (discounted_rewards.std() + eps)
    loss_inv = loss_inv_fun(torch.Tensor(acts), torch.stack(act_hats))
    loss_for = loss_for_fun(torch.stack(next_obs_lats).squeeze(0), torch.stack(next_obs_lat_hats).squeeze(0))
    loss_icm = loss_for + loss_for
    loss = [-advantage[i] * log_soft[i] for i in range(len(advantage))]
    loss = torch.stack(loss)
    loss.to(device)
    loss_icm.to(device)
    loss_icm.backward()
    loss.sum().backward()
    optimizer_icm.step()
    optimizer.step()
    wandb.log({
        "Episode reward": ep_reward,
        "Extrinsic reward": ex_reward,
        "Intrinsic reward": in_reward,
        "Loss": loss.cpu().mean(),
        "ICM Loss": loss_icm.cpu().mean()
        }, step=episode)

    if episode % 500 == 0 and episode != 0:
        visited_pos, visited_vel = visualize()
        plt.scatter(visited_pos, visited_vel, marker='.')
        plt.xlabel('position')
        plt.ylabel('velocity')
        wandb.log({
            "video": wandb.Video('/tmp/current_gif.gif', fps=4, format="gif")})

torch.save(vpg.state_dict(), "model.h5")
wandb.save('model.h5')

tot_per = []
epsilon = 0

