import random
import torch
import tqdm
import numpy as np


class ReplayBuffer():
    def __init__(self, Transition, size, env, batch_size, device):
        self.size = size
        self.buffer = []
        self.index = 0
        self.env = env
        self.Transition = Transition
        self.buffer_fields = self.Transition._fields
        self.batch_size = batch_size
        self.sample_size = self.index
        self.device = device

    def fill_buffer(self):
        obs = self.env.reset()
        done = False
        for trans in tqdm(range(0, self.size)):
            action = self.env.action_space.sample()
            new_obs, reward, done, _ = self.env.step(action)
            self.buffer.append(
                self.Transition(obs, action, reward, new_obs, done))
            if done:
                obs = self.env.reset()
                done = False
            else:
                obs = new_obs

    def store_filled(self, trans):
        self.index = (self.index + 1) % self.size
        self.buffer[self.index] = self.Transition(*trans)

    def store(self, trans):
        if (self.index + 1) % self.size:
            self.buffer.append(self.Transition(*trans))
            self.index += 1
        else:
            self.store_filled(trans)

    def extract_from_buffer(self):
        self.sampled_buffer = self.buffer
        self.sample_size = self.index
        for field in self.buffer_fields:
            name = field + 's'
            setattr(
                self, name,
                torch.stack([
                    getattr(trans, field) for trans in self.sampled_buffer
                ]).view(self.sample_size, -1))

    def gae(self, gamma, lambd=1):
        tds = self.rewards + gamma * self.next_values - self.values
        advantages = []
        ad = torch.zeros(1).to(self.device)
        for td in reversed(tds):
            ad = td + ad * gamma * lambd
            advantages.insert(0, ad)
        advantages = torch.stack(advantages)
        loss_cr = torch.pow(tds, 2)
        # TODO we cannot use episode returns since the sample is not in order.
        # so we use `tds` rather than `advatanges` at the moment
        return advantages, loss_cr

    def gae_losses(self, gamma, lambd=1):
        advantages, loss_cr = self.gae(gamma, lambd)
        loss_ac = advantages * self.log_probs
        return loss_ac, loss_cr

    def clipped_losses(self, new_log_probs, gamma, clip_rt, lambd=1):
        advantages, loss_cr = self.gae(gamma, lambd)
        ratio = torch.exp(new_log_probs - self.log_probs)
        clip_min = 1 - clip_rt
        clip_max = 1 + clip_rt
        clipped_ratio = torch.clamp(ratio, clip_min, clip_max)
        # clipped_adv = torch.max(torch.min(advantages, clip_max), clip_min)
        loss_ac = torch.min(clipped_ratio * advantages, ratio * advantages)
        return loss_ac, loss_cr

    def sample(self):
        self.sample_size = np.minimum(self.batch_size, self.index)
        return random.sample(self.buffer,
                             k=self.sample_size)

    def empty(self):
        self.buffer = []
        self.index = 0

    def get_list(self, item):
        return [getattr(trans, item) for trans in self.buffer]
