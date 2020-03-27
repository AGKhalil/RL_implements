import os
import numpy as np
import random
from tqdm import tqdm
import gym
import time
import copy
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple

import torch
import torch.tensor as Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.special import softmax

import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

import wandb
import imageio


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, env.action_space.n)
            
    def forward(self, x, softmax=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if softmax:
          return F.softmax(x)
        else:
          return x

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.index = 0
        
    def fill_buffer(self):
        obs = env.reset()
        done = False
        for trans in tqdm(range(0, self.size)):
            action = env.action_space.sample()
            new_obs, reward, done, _ = env.step(action)
            self.buffer.append(Transition(obs, action, reward, new_obs, done))
            if done:
                obs = env.reset()
                done = False
            else:
                obs = new_obs
    
    def store_filled(self, trans):
        self.index = (self.index + 1) % self.size
        self.buffer[self.index] = Transition(trans[0], trans[1], trans[2], trans[3], trans[4])
        
    def store(self, trans):
        if (self.index + 1) % self.size:
            self.buffer.append(Transition(trans[0], trans[1], trans[2], trans[3], trans[4]))
            self.index += 1
        else:
            self.store_filled(trans)
        
    def sample(self, batch=64):
        return random.sample(self.buffer, k=batch)

def get_action(obs):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return torch.argmax(get_current_value(obs)).item()
    
def get_target_value(obs):
    return target.forward(torch.from_numpy(obs).float().unsqueeze(0).to(gpu)).detach()

def get_current_value(obs):
    return value.forward(torch.from_numpy(obs).float().unsqueeze(0).to(gpu))

def get_softmax(obs):
  return value.forward(torch.from_numpy(obs).float().unsqueeze(0).to(gpu), softmax=True).detach()

def visualize():
    done = False
    obs = env.reset()
    img = env.render(mode='rgb_array')
    imgs = []
    while not done:
        imgs.append(img)
        act = get_action(obs)
        obs, rew, done, _ = env.step(act)
        img = env.render(mode='rgb_array')

    imageio.mimsave('/tmp/current_gif.gif', [np.array(img) for i, img in enumerate(imgs) if i%2 == 0], fps=29)       

env = gym.make('CartPole-v0')
buffer = ReplayBuffer(10000)
# buffer.fill_buffer()

wandb.init(entity="agkhalil", project="pytorch-vpg-mountaincar")
wandb.watch_called = False

config = wandb.config
config.batch_size = 64
config.episodes = 10000
config.lr = 1e-4
config.seed = 42
config.epsilon = 1
config.update_target = 500
config.gamma = 0.9
config.eps_start = 0.9
config.eps_end = 0.05
config.eps_decay = 0.999

gpu = torch.device('cpu')
torch.manual_seed(config.seed)
learning_rate = config.lr
batch_size = config.batch_size
value = MLP().to(gpu)
target = MLP().to(gpu)
optimizer = optim.Adam(value.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

EPISODES = config.episodes
update_target = config.update_target
epsilon = config.epsilon
gamma = config.gamma
eps_decay = config.eps_decay
eps_end = config.eps_end
rewards = []

wandb.watch(value, log="all")

for episode in tqdm(range(0, EPISODES)):
    obs = env.reset()
    done = False
    ep_reward = 0
    while not done:
        action = get_action(obs)
        new_obs, reward, done, _ = env.step(action)
        buffer.store((obs, action, torch.tensor(reward).to(gpu), new_obs, done))
        ep_reward += reward
        obs = new_obs

    if len(buffer.buffer) > batch_size:
        optimizer.zero_grad()
        minibatch = buffer.sample()
        next_qs = [i.reward if i.done else i.reward + gamma * get_target_value(i.next_state).max() for i in minibatch]
        current_qs = [get_current_value(i.state).squeeze(0)[i.action] for i in minibatch]
        current_qs_softmax = [get_softmax(i.state).squeeze(0) for i in minibatch]
        # current_qs_entropy = -np.sum(current_qs_softmax * np.log(current_qs_softmax)).cpu()
        # current_qs_entropy = current_qs_entropy[0] + current_qs_entropy[1]
        next_qs = torch.stack(next_qs)
        current_qs = torch.stack(current_qs)
        next_qs.to(gpu)
        current_qs.to(gpu)
        loss = loss_fn(current_qs, next_qs)
        loss.backward()
        optimizer.step()
        epsilon = max(epsilon*eps_decay,eps_end)
        wandb.log({
            "Episode reward": ep_reward,
            "Epsilon": epsilon,
            "Loss": loss,
            # "Sum of entropy over batch Q-values": current_qs_entropy
            }, step=episode)

    if episode % 100 == 0 and episode != 0:
        visualize()
        wandb.log({"video": wandb.Video('/tmp/current_gif.gif', fps=4, format="gif")})

    if episode % update_target == 0:
        target.load_state_dict(value.state_dict())

torch.save(value.state_dict(), "model.h5")
wandb.save('model.h5')