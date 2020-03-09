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
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.fc = nn.ModuleList()
        self.layers_size = len(layers)
        prev = layers[0]
        for n in range(1, self.layers_size):
            self.fc.append(nn.Linear(prev, layers[n]))
            prev = layers[n]
        self.layers_n = len(self.fc) - 1
            
    def forward(self, x):
        for n in range(self.layers_n):
            x = F.relu(self.fc[n](x))
        x = self.fc[self.layers_n](x)
        return x