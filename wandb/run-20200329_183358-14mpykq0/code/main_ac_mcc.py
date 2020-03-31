import numpy as np
from tqdm import tqdm
import math
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
import argparse

from sklearn import preprocessing
from bayes_opt import BayesianOptimization


def visualize(env, ac, cr):
    done = False
    obs = env.reset()
    imgs, visited_pos, visited_vel, acts, means, stds, vals = [], [], [], [], [], [], []
    #img = env.render('rgb_array')
    while not done:
    #    imgs.append(img)
        visited_pos.append(obs[0])
        visited_vel.append(obs[1])
        act, _ = ac.get_action(obs)
        val = cr.get_action(obs, critic=True)
        acts.append(act[0])
        vals.append(val.squeeze().detach().item())
        means.append(ac.action_mean.detach().item())
        stds.append(ac.action_std.detach().item())
        obs, rew, done, _ = env.step(act)
        #img = env.render('rgb_array')

    #imageio.mimsave('/tmp/current_gif.gif', [np.array(img) for i, img in enumerate(imgs) if i%2 == 0], fps=29)
    #optimize('/tmp/current_gif.gif')

    return visited_pos, visited_vel, acts, means, stds, vals

def evaluate(env, ac, cr):
    eval_rew = []
    for _ in tqdm(range(0, 100)):
        done = False
        obs = env.reset()
        ep_reward = 0
        while not done:
            act, _ = ac.get_action(obs)
            obs, rew, done, _ = env.step(act)
            ep_reward += rew
        eval_rew.append(ep_reward)

    return np.mean(eval_rew)

def net_layers(hidden, env_type, env):
    if env_type == 'DISCRETE':
        act_space = env.action_space.n
    else:
        act_space = env.action_space.shape[0]
    obs_space = env.observation_space.shape[0]
    return [obs_space] + hidden + [act_space]

def scale_state(obs, scaler):
    return scaler.transform([obs])

def main(lr_ac, lr_cr):
    wandb.init(entity="agkhalil", project="pytorch-ac-mountaincarcont-bayesopt2", reinit=True)
    wandb.watch_called = False

    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--lr_ac', type=float, default=0.001, metavar='lrac', help='actor learning rate')
    parser.add_argument('--lr_cr', type=float, default=0.000001, metavar='lrac',help='critic learning rate')
    args = parser.parse_args()

    config = wandb.config
    # config.update({"lr_ac": lr_ac, "lr_cr": lr_cr}, allow_val_change=True)
    config.batch_size = 50
    config.episodes = 500
    config.lr_ac = lr_ac
    config.lr_cr = lr_cr
    config.seed = 42
    config.gamma = 0.99
    eps = np.finfo(np.float32).eps.item()

    device = torch.device('cpu')
    torch.manual_seed(config.seed)
    lr_ac = config.lr_ac
    lr_cr = config.lr_cr
    batch_size = config.batch_size

    env = gym.make('MountainCarContinuous-v0')
    state_space_samples = np.array([env.observation_space.sample() for x in range(1000)])
    scaler = preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
    env_type = 'CONT'

    mlp_ac = MLP_AC(net_layers([32, 16], env_type, env)).to(device)
    mlp_cr = MLP_AC(net_layers([64, 32], env_type, env)).to(device)
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
        log_probs = []
        values_list = []
        next_values_list = []
        acts_list = []
        my_rew = []
        obs = env.reset()
        done = False
        ep_reward = 0
        step = 0
        while not done:
            action, log_prob = ac.get_action(scale_state(obs, scaler))
            acts_list.append(action)
            fuck = 0
            if done and step < 999:
               fuck = 100
            fuck -= math.pow(action[0], 2) * 0.1
            my_rew.append(fuck)
            value = cr.get_action(scale_state(obs, scaler), critic=True)
            new_obs, rew, done, _ = env.step(action)
            #print('ac, log, rew', action, log_prob, rew)
            next_value = cr.get_action(scale_state(new_obs, scaler), critic=True)
            ep_reward += rew
            rewards.append(rew)
            log_probs.append(log_prob)
            values_list.append(value)
            next_values_list.append(next_value)
            step += 1
            obs = new_obs

        print(np.sum(my_rew))

        rewards_size = len(rewards)
        gammas = [np.power(gamma, i) for i in range(rewards_size)]
        discounted_rewards = [np.sum(np.multiply(gammas[:rewards_size-i], rewards[i:])) for i in range(rewards_size)]
        discounted_rewards = torch.tensor(discounted_rewards).to(device)
        returns = [rewards[i] + gamma * next_values_list[i] for i in reversed(range(rewards_size))]
        # returns = torch.tensor(returns).to(device)
        # values_list = torch.tensor(values_list).to(device)
        # next_values_list = torch.tensor(next_values_list).to(device)

        # target = rew + gamma * next_value.detach()
        td = np.subtract(returns, values_list)
        values_list = torch.stack(values_list)
        returns = torch.stack(returns)
        loss_cr = loss_fn(values_list, returns)
        loss_ac = [-td[i].detach() * log_probs[i] for i in range(len(td))]
        loss_ac = torch.stack(loss_ac)
        # loss_ac.to(device)
        # loss_cr.to(device)
        
        optimizer_ac.zero_grad()
        optimizer_cr.zero_grad()

        # loss = loss_cr + loss_ac
        # print(action, loss_ac)
        
        loss_ac.sum().backward()
        loss_cr.backward()
        optimizer_cr.step()
        optimizer_ac.step()


        # for i in list(ac.policy.fc[0].parameters()):
        #     weird_sum += i.detach().sum().item()
        # print(weird_sum)

        # rewards_size = len(rewards)
        # gammas = [np.power(gamma, i) for i in range(rewards_size)]
        # discounted_rewards = [np.sum(np.multiply(gammas[:rewards_size-i], rewards[i:])) for i in range(rewards_size)]
        # optimizer.zero_grad()
        # discounted_rewards = torch.tensor(discounted_rewards).to(device)
        # advantage = discounted_rewards #- discounted_rewards.mean()) / (discounted_rewards.std() + eps)
        # loss = [-advantage[i] * log_probs[i] for i in range(len(advantage))]
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

        if episode % 500 == 0 and episode != 0:
            visited_pos, visited_vel, acts, means, stds, vals = visualize(env, ac, cr)
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
         #       "video": wandb.Video('/tmp/current_gif.gif', fps=4, format="gif"),
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
           # model_name = "model-" + str(episode) + ".h5"
           # torch.save(ac.policy.state_dict(), model_name)
           # wandb.save(model_name)
           # dir_path = os.path.dirname(os.path.realpath(__file__))
           # os.remove(dir_path + '/' + model_name)
    wandb.join()

    return evaluate(env, ac, cr)

if __name__ == "__main__":
    pbounds = {'lr_ac': (0.00001, 0.9), 'lr_cr': (0.00001, 0.9)}

    optimizer = BayesianOptimization(
        f=main,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=10,
    )

    print(optimizer.max)
