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


class VPG():
    def __init__(self, policy, env, device, env_type):
        self.device = device
        self.env = env
        self.policy = policy
        self.env_type = env_type

    def get_action(self, obs):
        if self.env_type == 'DISCRETE':
            current_policy = self.get_current_policy(obs)
            probs = F.softmax(current_policy)
            dist = torch.distributions.Categorical(probs)
            act = dist.sample().item()
            return act, F.log_softmax(current_policy).squeeze(0)[act]
        else:
            current_policy = self.get_current_policy(obs)
            dist = torch.distributions.normal.Normal(current_policy, 1)
            act = dist.sample().item()
            return [act], dist.log_prob(act)

    def get_current_policy(self, obs):
        return self.policy.forward(torch.from_numpy(obs).float().unsqueeze(0).to(self.device))