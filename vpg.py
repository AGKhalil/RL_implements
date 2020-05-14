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
from sklearn import preprocessing
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

import wandb
import imageio


class VPG():
    def __init__(self, policy, env, device, env_type):
        self.device = device
        self.env = env
        self.policy = policy
        self.env_type = env_type

    def get_action(self, obs):
        if self.env_type == 'DISCRETE':
            self.current_policy = self.get_current_policy(obs)
            self.probs = F.softmax(self.current_policy)
            self.dist = torch.distributions.Categorical(self.probs)
            self.act = self.dist.sample().item()
            return self.act, F.log_softmax(
                self.current_policy).squeeze(0)[self.act]
        else:
            self.current_policy = self.get_current_policy(obs)
            self.dist = torch.distributions.normal.Normal(
                self.current_policy, 1)
            self.act = self.dist.sample().squeeze()
            if self.act.nelement() == 1:
                return [self.act.item()], self.dist.log_prob(self.act)
            else:
                return self.act.numpy(), self.dist.log_prob(self.act)

    def get_current_policy(self, obs):
        return self.policy.forward(
            torch.from_numpy(obs).float().unsqueeze(0).to(self.device))


class AC():
    def __init__(self, policy, env, device, env_type):
        self.device = device
        self.env = env
        self.policy = policy
        self.env_type = env_type

    def get_action(self, obs, critic=False):
        if critic:
            return self.get_current_policy(obs, critic=critic)
        if self.env_type == 'DISCRETE':
            self.action_logit = self.get_current_policy(obs)
            self.probs = F.softmax(self.action_mean)
            self.dist = torch.distributions.Categorical(self.probs)
            self.act = self.dist.sample().item()
            return self.act, F.log_softmax(
                self.action_mean).squeeze(0)[self.act]
        else:
            self.action_mean, self.action_std = self.get_current_policy(obs)
            self.action_mean = self.action_mean.view(-1)
            self.action_std = self.action_std.view(-1)
            self.dist = torch.distributions.normal.Normal(
                self.action_mean, self.action_std + 1e-5)
            self.act = self.dist.sample().squeeze()
            if self.act.nelement() == 1:
                return [
                    torch.clamp(self.act, self.env.action_space.low[0],
                                self.env.action_space.high[0]).item()
                ], self.dist.log_prob(self.act)
            else:
                return self.act.numpy(), self.dist.log_prob(self.act)

    def get_current_policy(self, obs, critic=False):
        return self.policy.forward(
            obs, critic)
