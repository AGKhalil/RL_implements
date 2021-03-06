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

import os
import wandb
import imageio
from pygifsicle import optimize

from vpg import AC
from policies import MLP_AC


def visualize():
    done = False
    obs = env.reset()
    imgs, visited_pos, visited_vel, acts, means, stds, vals = [], [], [], [], [], [], []
    img = env.render('rgb_array')
    while not done:
        imgs.append(img)
        visited_pos.append(obs[0])
        visited_vel.append(obs[1])
        act, _ = ac.get_action(obs)
        val = cr.get_action(obs, critic=True)
        acts.append(act[0])
        vals.append(val.squeeze().detach().item())
        means.append(ac.action_mean.detach().item())
        stds.append(ac.action_std.detach().item())
        obs, rew, done, _ = env.step(act)
        img = env.render('rgb_array')

    imageio.mimsave('/tmp/current_gif.gif', [np.array(img) for i, img in enumerate(imgs) if i%2 == 0], fps=29)
    optimize('/tmp/current_gif.gif')

    return visited_pos, visited_vel, acts, means, stds, vals

def net_layers(hidden):
    if env_type == 'DISCRETE':
        act_space = env.action_space.n
    else:
        act_space = env.action_space.shape[0]
    obs_space = env.observation_space.shape[0]
    return [obs_space] + hidden + [act_space]


wandb.init(entity="agkhalil", project="pytorch-ac-mountaincarcont")
wandb.watch_called = False

config = wandb.config
config.batch_size = 50
config.episodes = 10000
config.lr_ac = 0.005
config.lr_cr = 0.00005
config.seed = 42
config.gamma = 0.99
eps = np.finfo(np.float32).eps.item()

device = torch.device('cpu')
torch.manual_seed(config.seed)
lr_ac = config.lr_ac
lr_cr = config.lr_cr
batch_size = config.batch_size

env = gym.make('MountainCarContinuous-v0')
env_type = 'CONT'

mlp_ac = MLP_AC(net_layers([32, 16])).to(device)
mlp_cr = MLP_AC(net_layers([64, 32])).to(device)
ac = AC(mlp_ac, env, device, env_type)
cr = AC(mlp_cr, env, device, env_type)
optimizer_cr = optim.Adam(cr.policy.parameters(), lr=lr_cr)
optimizer_ac = optim.Adam(ac.policy.parameters(), lr=lr_ac)
loss_fn = torch.nn.MSELoss()

EPISODES = config.episodes
gamma = config.gamma

wandb.watch(ac.policy, log="all")

visited_pos, visited_vel = [], []

for episode in tqdm(range(0, EPISODES)):
    rewards = []
    log_soft = []
    obs = env.reset()
    done = False
    ep_reward = 0
    step = 0
    while not done:
        action, log_prob = ac.get_action(obs)
        value = cr.get_action(obs, critic=True)
        new_obs, rew, done, _ = env.step(action)
        next_value = cr.get_action(new_obs, critic=True)
        target = rew + gamma * next_value
        td = rew + gamma * next_value.detach() - value.detach()
        loss_cr = loss_fn(value, target)
        loss_ac = -td * log_prob
        loss_ac.to(device)
        loss_cr.to(device)
        # rewards.append(rew)
        # log_soft.append(log_prob)
        optimizer_ac.zero_grad()
        optimizer_cr.zero_grad()
        # loss = loss_cr + loss_ac
        # print(action, loss_ac)
        
        loss_ac.backward()
        loss_cr.backward()
        optimizer_cr.step()
        optimizer_ac.step()

        ep_reward += rew
        step += 1
        obs = new_obs


    # weird_sum = 0
    # for i in list(ac.policy.fc[0].parameters()):
    #     weird_sum += i.detach().sum().item()
    # print(weird_sum)

    # rewards_size = len(rewards)
    # gammas = [np.power(gamma, i) for i in range(rewards_size)]
    # discounted_rewards = [np.sum(np.multiply(gammas[:rewards_size-i], rewards[i:])) for i in range(rewards_size)]
    # optimizer.zero_grad()
    # discounted_rewards = torch.tensor(discounted_rewards).to(device)
    # advantage = discounted_rewards #- discounted_rewards.mean()) / (discounted_rewards.std() + eps)
    # loss = [-advantage[i] * log_soft[i] for i in range(len(advantage))]
    # loss = torch.stack(loss)
    # loss.to(device)
    # loss.sum().backward()
    # optimizer.step()
    wandb.log({
        "Episode reward": ep_reward,
        "Episode length": step,
        "Policy Loss": loss_ac.cpu().mean(),
        "Value Loss": loss_cr.cpu().mean(),
        }, step=episode)

    if episode % 100 == 0 and episode != 0:
        visited_pos, visited_vel, acts, means, stds, vals = visualize()
        fig1 = plt.figure()
        plt.scatter(visited_pos, visited_vel, marker='.')
        plt.xlabel('position')
        plt.ylabel('velocity')
        
        fig2 = plt.figure()
        plt.scatter([i for i in range(len(acts))], acts, marker='.')
        plt.xlabel('steps')
        plt.ylabel('actions')
        
        fig3 = plt.figure()
        plt.scatter([i for i in range(len(means))], means, marker='.')
        plt.xlabel('steps')
        plt.ylabel('means')

        fig4 = plt.figure()
        plt.scatter([i for i in range(len(vals))], vals, marker='.')
        plt.xlabel('steps')
        plt.ylabel('values')

        fig5 = plt.figure()
        plt.scatter([i for i in range(len(stds))], stds, marker='.')
        plt.xlabel('steps')
        plt.ylabel('stds')
        
        wandb.log({
            "video": wandb.Video('/tmp/current_gif.gif', fps=4, format="gif"),
            "visited_pos": visited_pos,
            "visited_vel": visited_vel,
            "actions": acts,
            "means": means,
            "values": vals,
            "states": fig1,
            "actions/step": fig2,
            "means/step": fig3,
            "values/step": fig4,
            "stds/step": fig5,
            })
        model_name = "model-" + str(episode) + ".h5"
        torch.save(ac.policy.state_dict(), model_name)
        wandb.save(model_name)
        os.remove(os.path.dirname("/home/oe18433/code_bases/RL_implements/") + '/' + model_name)