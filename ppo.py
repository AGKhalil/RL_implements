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
from utils import env_wandb, evaluate, net_layers, tensor_obs
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)


def main(lr_ac, lr_cr, clip_rt, lambd):
    wandb.init(entity="agkhalil",
               project="pytorch-ppo-lunar-bayesopt1",
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
    config.batch_size = 64
    config.episodes = 5000
    config.lr_ac = lr_ac
    config.lr_cr = lr_cr
    config.seed = 42
    config.gamma = 0.99
    config.clip_rt = clip_rt
    config.lambd = lambd

    device = torch.device('cpu')
    torch.manual_seed(config.seed)

    env = gym.make('LunarLanderContinuous-v2')
    env_type = 'CONT'

    mlp_ac = MLP_AC(net_layers([32, 16], env_type, env)).to(device)
    mlp_cr = MLP_AC(net_layers([64, 32], env_type, env)).to(device)
    ac = AC(mlp_ac, env, device, env_type)
    new_ac = AC(mlp_ac, env, device, env_type)
    cr = AC(mlp_cr, env, device, env_type)
    optimizer_cr = optim.Adam(cr.policy.parameters(), lr=config.lr_cr)
    optimizer_ac = optim.Adam(new_ac.policy.parameters(), lr=config.lr_ac)

    Transition = namedtuple('Transition',
                            ('state', 'action', 'log_prob', 'reward',
                             'next_state', 'done', 'value', 'next_value'))
    r_buffer = ReplayBuffer(Transition=Transition,
                            size=10000,
                            env=env,
                            batch_size=config.batch_size)

    wandb.watch(ac.policy, log="all")

    for episode in tqdm(range(0, config.episodes)):
        ac.policy.load_state_dict(new_ac.policy.state_dict())
        obs = tensor_obs(env.reset()).to(device)
        done = False
        ep_reward = 0
        step = 0
        while not done:
            action, log_prob = ac.get_action(obs)
            value = cr.get_action(obs, critic=True)
            new_obs, rew, done, _ = env.step(action)
            new_obs = tensor_obs(new_obs)
            rew = torch.tensor(rew).type(torch.FloatTensor).detach()
            next_value = cr.get_action(new_obs, critic=True)
            trans = (obs, torch.tensor(action), log_prob.squeeze(), rew,
                     new_obs, torch.tensor(done).detach(), value.squeeze(),
                     next_value.squeeze())
            r_buffer.store(trans)

            ep_reward += rew
            step += 1
            obs = new_obs

        r_buffer.extract_from_buffer()
        for i in range(16):
            new_log_probs = new_ac.get_action(r_buffer.states)[-1].view(
                r_buffer.sample_size, -1)
            loss_ac, loss_cr = r_buffer.clipped_losses(new_log_probs,
                                                       gamma=config.gamma,
                                                       clip_rt=config.clip_rt,
                                                       lambd=config.lambd)
            optimizer_ac.zero_grad()
            # loss_ac = (loss_ac - loss_ac.mean()) / (loss_ac.std() + 1e-8)
            loss_ac = -loss_ac.mean()
            loss_ac.sum().backward(retain_graph=True)
            optimizer_ac.step()
        optimizer_cr.zero_grad()
        loss_cr.mean().backward()
        optimizer_cr.step()
        r_buffer.empty()

        wandb.log(
            {
                "Episode reward": ep_reward,
                "Episode length": step,
                "Policy Loss": loss_ac.cpu().mean(),
                "Value Loss": loss_cr.cpu().mean(),
            },
            step=episode)

        if episode % config.episodes == 0 and episode != 0:
            env_wandb(env, ac, cr, wandb)
    wandb.join()

    return evaluate(env, ac, cr)


if __name__ == "__main__":
    pbounds = {
        'lr_ac': (1e-5, 1e-1),
        'lr_cr': (1e-5, 1e-1),
        'clip_rt': (0.01, 0.3),
        'lambd': (0.8, 1)
    }

    optimizer = BayesianOptimization(
        f=main,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=50,
    )

    print(optimizer.max)
