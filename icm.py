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
from math import floor

class ICM(nn.Module):
    def __init__(self, env, device):
        super(ICM, self).__init__()
        self.obs_latent_shape = 128
        self.env = env
        self.device = device
        self.obs_space = self.env.observation_space.shape[0]
        self.act_space = self.env.action_space.shape[0]
        self.encoder = nn.Sequential(
                    nn.Linear(self.obs_space, 128),
                    nn.ELU(inplace=True),
                    nn.Linear(128, 128),
                    nn.ELU(inplace=True),
                    nn.Linear(128, self.obs_latent_shape),
        )
        self.inverse_model = nn.Sequential(
                    nn.Linear(self.obs_latent_shape * 2, 128),
                    nn.ELU(inplace=True),
                    nn.Linear(128, 128),
                    nn.ELU(inplace=True),
                    nn.Linear(128, self.act_space),
        )
        self.forward_model = nn.Sequential(
                    nn.Linear(128 + self.act_space, 128),
                    nn.ELU(inplace=True),
                    nn.Linear(128, 128),
                    nn.ELU(inplace=True),
                    nn.Linear(128, self.obs_latent_shape),
        )

    def forward(self, obs, next_obs):
        self.obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        self.next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).to(self.device)
        self.obs_latent = self.encoder(self.obs)
        self.next_obs_latent = self.encoder(self.next_obs)
        self.inv_input = torch.cat((self.obs_latent, self.next_obs_latent), dim=1)
        self.action_hat = self.inverse_model(self.inv_input)
        self.for_input = torch.cat((self.action_hat, self.obs_latent), dim=1)
        self.next_obs_latent_hat = self.forward_model(self.for_input)
        return self.next_obs_latent, self.next_obs_latent, self.next_obs_latent_hat, self.action_hat
