import argparse
import logging

import gym
import numpy as np
import torch
import torch.optim as optim
from bayes_opt import BayesianOptimization
from sklearn import preprocessing
from tqdm import tqdm
from collections import namedtuple

import wandb
from policies import MLP_AC
from vpg import AC
from replay_buffer import ReplayBuffer
from utils import env_wandb, evaluate, net_layers, scale_state
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)


def main(lr_ac=0.4979, lr_cr=0.6318):
    wandb.init(entity="agkhalil",
               project="pytorch-ac-mountaincarcont-bayesopt5",
               reinit=True)
    wandb.watch_called = False

    parser = argparse.ArgumentParser(
        description='PyTorch actor-critic example')
    parser.add_argument('--lr_ac',
                        type=float,
                        default=0.1321,
                        help='actor learning rate')
    parser.add_argument('--lr_cr',
                        type=float,
                        default=0.08311,
                        help='critic learning rate')
    args = parser.parse_args()

    config = wandb.config
    config.batch_size = 50
    config.episodes = 500
    config.lr_ac = lr_ac
    config.lr_cr = lr_cr
    config.seed = 42
    config.gamma = 0.99

    device = torch.device('cpu')
    torch.manual_seed(config.seed)
    lr_ac = config.lr_ac
    lr_cr = config.lr_cr

    env = gym.make('MountainCarContinuous-v0')
    state_space_samples = np.array(
        [env.observation_space.sample() for x in range(1000)])
    scaler = preprocessing.StandardScaler()
    scaler.fit(state_space_samples)
    env_type = 'CONT'

    mlp_ac = MLP_AC(net_layers([32, 16], env_type, env)).to(device)
    mlp_cr = MLP_AC(net_layers([64, 32], env_type, env)).to(device)
    ac = AC(mlp_ac, env, device, env_type)
    cr = AC(mlp_cr, env, device, env_type)
    optimizer_cr = optim.Adam(cr.policy.parameters(), lr=lr_cr)
    optimizer_ac = optim.Adam(ac.policy.parameters(), lr=lr_ac)

    EPISODES = config.episodes
    gamma = config.gamma

    Transition = namedtuple('Transition',
                            ('state', 'action', 'log_prob', 'reward',
                             'next_state', 'done', 'value', 'next_value'))
    r_buffer = ReplayBuffer(Transition=Transition, size=10000, env=env)

    wandb.watch(ac.policy, log="all")

    for episode in tqdm(range(0, EPISODES)):
        rewards = []
        log_probs = []
        values_list = []
        next_values_list = []
        acts_list = []
        obs = env.reset()
        done = False
        ep_reward = 0
        step = 0
        while not done:
            action, log_prob = ac.get_action(scale_state(obs, scaler))
            value = cr.get_action(scale_state(obs, scaler), critic=True)
            new_obs, rew, done, _ = env.step(action)
            next_value = cr.get_action(scale_state(new_obs, scaler),
                                       critic=True)
            trans = (obs, action, log_prob.squeeze(), rew, new_obs, done,
                     value.squeeze(), next_value.squeeze())
            r_buffer.store(trans)

            ep_reward += rew
            acts_list.append(action)
            rewards.append(rew)
            log_probs.append(log_prob)
            values_list.append(value)
            next_values_list.append(next_value)
            step += 1
            obs = new_obs

        # rewards_size = len(rewards)
        # gammas = [np.power(gamma, i) for i in range(rewards_size)]
        # discounted_rewards = [
        # np.sum(np.multiply(gammas[:rewards_size - i], rewards[i:]))
        # for i in range(rewards_size)
        # ]
        # discounted_rewards = torch.tensor(discounted_rewards).to(device)
        # returns = [
        # rewards[i] + gamma * next_values_list[i]
        # for i in reversed(range(rewards_size))
        # ]

        # td = np.subtract(returns, values_list)
        # values_list = torch.stack(values_list)
        # returns = torch.stack(returns)
        # loss_cr = loss_fn(values_list, returns)
        # loss_ac = [-td[i].detach() * log_probs[i] for i in range(len(td))]
        # loss_ac = torch.stack(loss_ac)

        loss_ac, loss_cr = r_buffer.get_losses_offline(gamma=gamma)
        optimizer_ac.zero_grad()
        optimizer_cr.zero_grad()
        loss_cr = torch.stack(loss_cr)
        loss_ac = torch.stack(loss_ac)
        loss_cr.mean().backward(retain_graph=True)
        loss_ac.sum().backward()
        optimizer_cr.step()
        optimizer_ac.step()
        r_buffer.empty()

        wandb.log(
            {
                "Episode reward": ep_reward,
                "Episode length": step,
                "Policy Loss": loss_ac.cpu().mean(),
                "Value Loss": loss_cr.cpu().mean(),
            },
            step=episode)

        if episode % 500 == 0 and episode != 0:
            env_wandb(env, ac, cr, wandb)
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
        n_iter=20,
    )

    print(optimizer.max)
