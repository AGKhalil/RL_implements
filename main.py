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

from vpg import VPG
from policies import MLP


def visualize():
    done = False
    obs = env.reset()
    visited_pos, visited_vel = [], []
    #img = env.render('rgb_array')
    while not done:
        #imgs.append(img)
        visited_pos.append(obs[0])
        visited_vel.append(obs[1])
        act, _ = vpg.get_action(obs)
        obs, rew, done, _ = env.step(act)
        #img = env.render('rgb_array')

    #imageio.mimsave('/tmp/current_gif.gif', [np.array(img) for i, img in enumerate(imgs) if i%2 == 0], fps=29)

    return visited_pos, visited_vel

def net_layers():
    if env_type == 'DISCRETE':
        act_space = env.action_space.n
    else:
        act_space = env.action_space.shape[0]
    obs_space = env.observation_space.shape[0]
    return [obs_space, 32, 16, act_space]


wandb.init(entity="agkhalil", project="pytorch-vpg-mountaincarcont")
wandb.watch_called = False

config = wandb.config
config.batch_size = 50
config.episodes = 30000
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
optimizer = optim.Adam(vpg.policy.parameters(), lr=learning_rate)

EPISODES = config.episodes
gamma = config.gamma

wandb.watch(vpg.policy, log="all")

visited_pos, visited_vel = [], []

for episode in tqdm(range(0, EPISODES)):
    rewards = []
    log_soft = []
    obs = env.reset()
    done = False
    ep_reward = 0
    reward = 0
    while not done:
        action, log_prob = vpg.get_action(obs)
        new_obs, rew, done, _ = env.step(action)
        rewards.append(rew)
        log_soft.append(log_prob)
        ep_reward += rew
        obs = new_obs

    rewards_size = len(rewards)
    gammas = [np.power(gamma, i) for i in range(rewards_size)]
    discounted_rewards = [np.sum(np.multiply(gammas[:rewards_size-i], rewards[i:])) for i in range(rewards_size)]
    optimizer.zero_grad()
    discounted_rewards = torch.tensor(discounted_rewards).to(device)
    advantage = discounted_rewards #- discounted_rewards.mean()) / (discounted_rewards.std() + eps)
    loss = [-advantage[i] * log_soft[i] for i in range(len(advantage))]
    loss = torch.stack(loss)
    loss.to(device)
    loss.sum().backward()
    optimizer.step()
    wandb.log({
        "Episode reward": ep_reward,
        "Loss": loss.cpu().mean(),
        }, step=episode)

    if episode % 500 == 0 and episode != 0:
        visited_pos, visited_vel = visualize()
        plt.scatter(visited_pos, visited_vel, marker='.')
        plt.xlabel('position')
        plt.ylabel('velocity')
        #wandb.log({
            #"video": wandb.Video('/tmp/current_gif.gif', fps=4, format="gif")})

#torch.save(vpg.state_dict(), "model.h5")
#wandb.save('model.h5')
