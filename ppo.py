import argparse
import logging

import gym
import optuna
import torch
import torch.optim as optim
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from collections import namedtuple

import wandb
from policies import MLP_AC
from vpg import AC
from replay_buffer import ReplayBuffer
from utils import env_wandb, evaluate, net_layers, tensor_obs
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)


def main(trial):
    wandb.init(entity="agkhalil",
               project="pytorch-ppo-lunar-bayesopt1",
               reinit=True)
    wandb.watch_called = False
    config = wandb.config
    config.batch_size = trial.suggest_int('batch_size', 16, 264)
    config.episodes = 5000
    config.lr_ac = trial.suggest_loguniform('lr_ac', 2e-6, 2e-1)
    config.lr_cr = trial.suggest_loguniform('lr_cr', 2e-6, 2e-1)
    config.seed = 42
    config.gamma = 0.99
    config.clip_rt = trial.suggest_uniform('clip_rt', 1e-2, 3e-1)
    config.lambd = trial.suggest_uniform('lambd', 0.8, 1.0)
    config.buffer_size = trial.suggest_int('buffer_size', 1e3, 5e4)
    config.epochs = trial.suggest_int('epochs', 1, 50)
    config.n_layers = trial.suggest_int('n_layers', 1, 3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    env = gym.make('LunarLanderContinuous-v2')
    env_type = 'CONT'

    mlp_ac = MLP_AC(net_layers([32] * config.n_layers, env_type,
                               env)).to(device)
    mlp_cr = MLP_AC(net_layers([32] * config.n_layers, env_type,
                               env)).to(device)
    ac = AC(mlp_ac, env, device, env_type)
    new_ac = AC(mlp_ac, env, device, env_type)
    cr = AC(mlp_cr, env, device, env_type)
    optimizer_cr = optim.Adam(cr.policy.parameters(), lr=config.lr_cr)
    optimizer_ac = optim.Adam(new_ac.policy.parameters(), lr=config.lr_ac)

    Transition = namedtuple('Transition',
                            ('state', 'action', 'log_prob', 'reward',
                             'next_state', 'done', 'value', 'next_value'))
    r_buffer = ReplayBuffer(Transition=Transition,
                            size=config.buffer_size,
                            env=env,
                            batch_size=config.batch_size,
                            device=device)

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
            new_obs = tensor_obs(new_obs).to(device)
            rew = torch.tensor(rew).type(torch.FloatTensor).detach()
            next_value = cr.get_action(new_obs, critic=True)
            trans = (obs, torch.tensor(action).to(device), log_prob.squeeze(),
                     rew.to(device), new_obs, torch.tensor(done).detach(),
                     value.squeeze(), next_value.squeeze())
            r_buffer.store(trans)

            ep_reward += rew
            step += 1
            obs = new_obs

        r_buffer.extract_from_buffer()
        optimizer_cr.zero_grad()
        for i in range(config.epochs):
            optimizer_ac.zero_grad()
            new_log_probs = new_ac.get_action(
                r_buffer.states,
                action=r_buffer.actions)[-1].view(r_buffer.sample_size, -1)
            loss_ac, loss_cr = r_buffer.clipped_losses(new_log_probs,
                                                       gamma=config.gamma,
                                                       clip_rt=config.clip_rt,
                                                       lambd=config.lambd)
            loss_ac = (loss_ac - loss_ac.mean()) / (loss_ac.std() + 1e-8)
            loss_ac.sum().backward(retain_graph=True)
            optimizer_ac.step()
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

    return evaluate(env, ac, cr, device)


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(main, n_trials=100)
    study.best_params
