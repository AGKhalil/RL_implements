import random
import tqdm
import numpy as np


class ReplayBuffer():
    def __init__(self, Transition, size, env):
        self.size = size
        self.buffer = []
        self.index = 0
        self.env = env
        self.Transition = Transition

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

    def get_losses_offline(self, gamma, lambd=1):
        tds = [
            trans.reward + gamma * trans.next_value - trans.value
            for trans in self.buffer
        ]
        discounts = [np.power(gamma * lambd, i) for i in range(self.index)]
        advantages = [
            np.sum(np.multiply(discounts[:self.index - i], tds[i:]))
            for i in range(self.index)
        ]
        loss_cr = np.power(tds, 2).tolist()
        loss_ac = np.multiply(advantages,
                              [trans.log_prob
                               for trans in self.buffer]).tolist()
        return loss_ac, loss_cr

    def sample(self, batch=64):
        return random.sample(self.buffer, k=batch)

    def empty(self):
        self.buffer = []
        self.index = 0
