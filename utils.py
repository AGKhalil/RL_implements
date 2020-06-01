import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from pygifsicle import optimize
from tqdm import tqdm


def tensor_obs(obs):
    return torch.tensor(obs).type(torch.FloatTensor).view(1, -1).detach()


def evaluate(env, ac, cr, device):
    eval_rew = []
    for _ in tqdm(range(0, 100)):
        done = False
        obs = tensor_obs(env.reset()).to(device)
        ep_reward = 0
        while not done:
            act, _ = ac.get_action(obs)
            obs, rew, done, _ = env.step(act)
            obs = tensor_obs(obs).to(device)
            ep_reward += rew
        eval_rew.append(ep_reward)

    return np.sum(eval_rew)


def visualize(env, ac, cr=None, gif=False):
    done = False
    obs = tensor_obs(env.reset())
    imgs, visited_pos, visited_vel = [], [], [],
    acts, means, stds, vals = [], [], [], []
    if gif:
        img = env.render('rgb_array')
    while not done:
        if gif:
            imgs.append(img)
        visited_pos.append(obs[0].detach().item())
        visited_vel.append(obs[1].detach().item())
        act, _ = ac.get_action(obs)
        if cr:
            val = cr.get_action(obs, critic=True)
            vals.append(val.squeeze().detach().item())
        acts.append(act[0])
        means.append(ac.action_mean.detach().item())
        stds.append(ac.action_std.detach().item())
        obs, rew, done, _ = env.step(act)
        obs = tensor_obs(obs)
        if gif:
            img = env.render('rgb_array')

    if gif:
        imageio.mimsave(
            '/tmp/current_gif.gif',
            [np.array(img) for i, img in enumerate(imgs) if i % 2 == 0],
            fps=29)
        optimize('/tmp/current_gif.gif')

    return visited_pos, visited_vel, acts, means, stds, vals


def env_wandb(env, ac, cr, wandb, gif=False):
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

    if gif:
        wandb.log({
            "video":
            wandb.Video('/tmp/current_gif.gif', fps=4, format="gif"),
            "visited_pos":
            visited_pos,
            "visited_vel":
            visited_vel,
            "actions":
            acts,
            "means":
            means,
            "values":
            vals,
            "states":
            fig1,
            "actions/step":
            fig2,
            "means/step":
            fig3,
            "values/step":
            fig4,
            "stds/step":
            fig5,
        })
    else:
        wandb.log({
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


def net_layers(hidden, env_type, env):
    if env_type == 'DISCRETE':
        act_space = env.action_space.n
    else:
        act_space = env.action_space.shape[0]
    obs_space = env.observation_space.shape[0]
    return [obs_space] + hidden + [act_space]
