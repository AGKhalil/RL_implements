import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna


# Behavior Function
class BF(nn.Module):
    def __init__(self, state_space, action_space, hidden_size, seed,
                 horizon_scale, return_scale):
        super(BF, self).__init__()
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale
        torch.manual_seed(seed)
        self.actions = np.arange(action_space)
        self.action_space = action_space
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.commands = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, command):
        out = self.sigmoid(self.fc1(state))
        command_out = self.sigmoid(self.commands(command))
        out = out * command_out
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def action(self, state, desire, horizon):
        """
        Samples the action based on their probability
        """
        command = torch.cat(
            (desire * self.return_scale, horizon * self.horizon_scale), dim=-1)
        action_prob = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action

    def greedy_action(self, state, desire, horizon):
        """
        Returns the greedy action
        """
        command = torch.cat(
            (desire * self.return_scale, horizon * self.horizon_scale), dim=-1)
        action_prob = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        action = torch.argmax(probs).item()
        return action


# Replay Buffer
class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add_sample(self, states, actions, rewards):
        episode = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "summed_rewards": sum(rewards)
        }
        self.buffer.append(episode)

    def sort(self):
        # sort buffer
        self.buffer = sorted(self.buffer,
                             key=lambda i: i["summed_rewards"],
                             reverse=True)
        # keep the max buffer size
        self.buffer = self.buffer[:self.max_size]

    def get_random_samples(self, batch_size):
        self.sort()
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return batch

    def get_nbest(self, n):
        self.sort()
        return self.buffer[:n]

    def __len__(self):
        return len(self.buffer)


class UDRL():
    def __init__(self, env, action_space, state_space, max_reward,
                 max_episodes, device, replay_size, n_warm_up_episodes,
                 n_updates_per_iter, n_episodes_per_iter, last_few, batch_size,
                 init_desired_reward, init_time_horizon, buffer, bf, optimizer,
                 return_scale, horizon_scale):
        self.last_few = last_few
        self.buffer = buffer
        self.state_space = state_space
        self.env = env
        self.action_space = action_space
        self.max_reward = max_reward
        self.max_episodes = max_episodes
        self.device = device
        self.replay_size = replay_size
        self.n_warm_up_episodes = n_warm_up_episodes
        self.n_updates_per_iter = n_updates_per_iter
        self.n_episodes_per_iter = n_episodes_per_iter
        self.batch_size = batch_size
        self.init_desired_reward = init_desired_reward
        self.init_time_horizon = init_time_horizon
        self.buffer = buffer
        self.bf = bf
        self.optimizer = optimizer
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale

    def sampling_exploration(self):
        """
        This function calculates the new desired reward and new desired horizon
        based on the replay buffer. New desired horizon is calculted by the
        mean length of the best last X episodes. New desired reward is sampled
        from a uniform distribution given the mean and the std calculated from
        the last best X performances. where X is the hyperparameter last_few.
        """

        top_X = self.buffer.get_nbest(self.last_few)
        # The exploratory desired horizon dh0 is set to the mean of the lengths
        # of the selected episodes
        new_desired_horizon = np.mean([len(i["states"]) for i in top_X])
        # save all top_X cumulative returns in a list
        returns = [i["summed_rewards"] for i in top_X]
        # from these returns calc the mean and std
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        # sample desired reward from a uniform distribution given the mean and
        # the std
        new_desired_reward = np.random.uniform(mean_returns,
                                               mean_returns + std_returns)

        return torch.FloatTensor([new_desired_reward
                                  ]), torch.FloatTensor([new_desired_horizon])

    # FUNCTIONS FOR TRAINING
    def select_time_steps(self, saved_episode):
        """
        Given a saved episode from the replay buffer this function samples
        random time steps (t1 and t2) in that episode:
        T = max time horizon in that episode
        Returns t1, t2 and T
        """
        # Select times in the episode:
        T = len(saved_episode["states"])  # episode max horizon
        t1 = np.random.randint(0, T - 1)
        t2 = np.random.randint(t1 + 1, T)

        return t1, t2, T

    def create_training_input(self, episode, t1, t2):
        """
        Based on the selected episode and the given time steps this function
        returns 4 values:
        1. state at t1
        2. the desired reward: sum over all rewards from t1 to t2
        3. the time horizont: t2 -t1
        4. the target action taken at t1
        buffer episodes are build like [cumulative episode reward, states,
        actions, rewards]
        """
        state = episode["states"][t1]
        desired_reward = sum(episode["rewards"][t1:t2])
        time_horizont = t2 - t1
        action = episode["actions"][t1]
        return state, desired_reward, time_horizont, action

    def create_training_examples(self, batch_size):
        """
        Creates a data set of training examples that can be used to create a
        data loader for training.
        ============================================================
        1. for the given batch_size episode idx are randomly selected
        2. based on these episodes t1 and t2 are samples for each selected
        episode
        3. for the selected episode and sampled t1 and t2 trainings values are
        gathered
        ______________________________________________________________
        Output are two numpy arrays in the length of batch size:
        Input Array for the Behavior function - consisting of (state,
        desired_reward, time_horizon)
        Output Array with the taken actions
        """
        input_array = []
        output_array = []
        # select randomly episodes from the buffer
        episodes = self.buffer.get_random_samples(batch_size)
        for ep in episodes:
            # select time stamps
            t1, t2, T = self.select_time_steps(ep)
            # For episodic tasks they set t2 to T:
            t2 = T
            state, desired_reward, time_horizont, action = self.create_training_input(
                ep, t1, t2)
            input_array.append(
                torch.cat([
                    state,
                    torch.FloatTensor([desired_reward]),
                    torch.FloatTensor([time_horizont])
                ]))
            output_array.append(action)
        return input_array, output_array

    def train_behavior_function(self, batch_size):
        """
        Trains the BF with on a cross entropy loss were the inputs are the
        action probabilities based on the state and command. The targets are
        the actions appropriate to the states from the replay buffer.
        """
        X, y = self.create_training_examples(batch_size)

        X = torch.stack(X)
        state = X[:, 0:self.state_space]
        d = X[:, self.state_space:self.state_space + 1]
        h = X[:, self.state_space + 1:self.state_space + 2]
        command = torch.cat((d * self.return_scale, h * self.horizon_scale),
                            dim=-1)
        y = torch.stack(y).long()
        y_ = self.bf(state.to(self.device), command.to(self.device)).float()
        self.optimizer.zero_grad()
        pred_loss = F.cross_entropy(y_, y)
        pred_loss.backward()
        self.optimizer.step()
        return pred_loss.detach().cpu().numpy()

    def evaluate(self, desired_return, desired_time_horizon):
        """
        Runs one episode of the environment to evaluate the bf.
        """
        state = self.env.reset()
        rewards = 0
        while True:
            state = torch.FloatTensor(state)
            action = self.bf.action(state.to(self.device),
                                    desired_return.to(self.device),
                                    desired_time_horizon.to(self.device))
            state, reward, done, _ = self.env.step(action.cpu().numpy())
            rewards += reward
            desired_return = min(desired_return - reward,
                                 torch.FloatTensor([self.max_reward]))
            desired_time_horizon = max(desired_time_horizon - 1,
                                       torch.FloatTensor([1]))

            if done:
                break
        return rewards

    # Training Loop
    # Algorithm 2 - Generates an Episode unsing the Behavior Function:
    def generate_episode(self, desired_return, desired_time_horizon):
        """
        Generates more samples for the replay buffer.
        """
        state = self.env.reset()
        states = []
        actions = []
        rewards = []
        while True:
            state = torch.FloatTensor(state)

            action = self.bf.action(state.to(self.device),
                                    desired_return.to(self.device),
                                    desired_time_horizon.to(self.device))
            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            desired_return -= reward
            desired_time_horizon -= 1
            desired_time_horizon = torch.FloatTensor(
                [np.maximum(desired_time_horizon, 1).item()])

            if done:
                break
        return [states, actions, rewards]

    # Algorithm 1 - Upside - Down Reinforcement Learning
    def run_upside_down(self, wandb, max_episodes):
        """
        """
        all_rewards = []
        losses = []
        average_100_reward = []
        desired_rewards_history = []
        horizon_history = []
        for ep in range(1, max_episodes + 1):

            # improve|optimize bf based on replay buffer
            loss_buffer = []
            for i in range(self.n_updates_per_iter):
                bf_loss = self.train_behavior_function(self.batch_size)
                loss_buffer.append(bf_loss)
            bf_loss = np.mean(loss_buffer)
            losses.append(bf_loss)

            # run x new episode and add to buffer
            for i in range(self.n_episodes_per_iter):

                # Sample exploratory commands based on buffer
                new_desired_reward, new_desired_horizon = self.sampling_exploration(
                )
                generated_episode = self.generate_episode(
                    new_desired_reward, new_desired_horizon)
                self.buffer.add_sample(generated_episode[0],
                                       generated_episode[1],
                                       generated_episode[2])

            new_desired_reward, new_desired_horizon = self.sampling_exploration(
            )
            # monitoring desired reward and desired horizon
            desired_rewards_history.append(new_desired_reward.item())
            horizon_history.append(new_desired_horizon.item())

            ep_rewards = self.evaluate(new_desired_reward, new_desired_horizon)
            all_rewards.append(ep_rewards)
            average_100_reward.append(np.mean(all_rewards[-100:]))

            wandb.log(
                {
                    "Episode reward": ep_rewards,
                    "Average reward (100)": np.mean(all_rewards[-100:]),
                    "Desired horizon": new_desired_horizon.item(),
                    "Desired reward": new_desired_reward.item(),
                    "BF loss": bf_loss
                },
                step=ep)

            print(
                "\rEpisode: {} | Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f}"
                .format(ep, ep_rewards, np.mean(all_rewards[-100:]), bf_loss),
                end="",
                flush=True)
            if ep % 100 == 0:
                print(
                    "\rEpisode: {} | Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f}"
                    .format(ep, ep_rewards, np.mean(all_rewards[-100:]),
                            bf_loss))

        return all_rewards, average_100_reward, desired_rewards_history, horizon_history, losses

    def warmup(self):
        # initial command
        for i in range(self.n_warm_up_episodes):
            desired_return = torch.FloatTensor([self.init_desired_reward])
            desired_time_horizon = torch.FloatTensor([self.init_time_horizon])
            state = self.env.reset()
            states = []
            actions = []
            rewards = []
            while True:
                action = self.bf.action(
                    torch.from_numpy(state).float().to(self.device),
                    desired_return.to(self.device),
                    desired_time_horizon.to(self.device))
                next_state, reward, done, _ = self.env.step(
                    action.cpu().numpy())
                states.append(torch.from_numpy(state).float())
                actions.append(action)
                rewards.append(reward)

                state = next_state
                desired_return -= reward
                desired_time_horizon -= 1
                desired_time_horizon = torch.FloatTensor(
                    [np.maximum(desired_time_horizon, 1).item()])

                if done:
                    break

            self.buffer.add_sample(states, actions, rewards)


def main(trial):
    # init Environment
    wandb.init(entity="agkhalil", project="pytorch-udrl-cartpole", reinit=True)
    wandb.watch_called = False
    config = wandb.config
    env = gym.make("CartPole-v0")
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    config.max_reward = 200
    config.max_episodes = 200
    config.device = torch.device(
        "cuda:1" if torch.cuda.is_available() else "cpu")
    config.horizon_scale = trial.suggest_categorical(
        'horizon_scale', [0.01, 0.015, 0.02, 0.025, 0.03])
    config.return_scale = trial.suggest_categorical(
        'return_scale', [0.01, 0.015, 0.02, 0.025, 0.03])
    config.replay_size = trial.suggest_categorical('replay_size',
                                                   [300, 400, 500, 600, 700])
    config.n_warm_up_episodes = trial.suggest_categorical(
        'n_warm_up_episodes', [10, 30, 50])
    config.n_updates_per_iter = trial.suggest_categorical(
        'n_updates_per_iter', [100, 150, 200, 250, 300])
    config.n_episodes_per_iter = trial.suggest_categorical(
        'n_episodes_per_iter', [10, 20, 30, 40])
    config.last_few = trial.suggest_categorical('last_few', [25, 50, 75, 100])
    config.batch_size = trial.suggest_categorical('batch_size',
                                                  [512, 768, 1024, 1536, 2048])
    config.init_desired_reward = 1
    config.init_time_horizon = 1

    # init replay buffer with n-warmup runs
    buffer = ReplayBuffer(config.replay_size)
    bf = BF(state_space, action_space, 64, 1, config.horizon_scale,
            config.return_scale).to(config.device)
    optimizer = optim.Adam(params=bf.parameters(), lr=1e-3)
    udrl = UDRL(env, action_space, state_space, config.max_reward,
                config.max_episodes, config.device, config.replay_size,
                config.n_warm_up_episodes, config.n_updates_per_iter,
                config.n_episodes_per_iter, config.last_few, config.batch_size,
                config.init_desired_reward, config.init_time_horizon, buffer,
                bf, optimizer, config.return_scale, config.horizon_scale)
    udrl.warmup()
    rewards, average, d, h, loss = udrl.run_upside_down(
        wandb, max_episodes=config.max_episodes)

    # SAVE MODEL
    name = "model.pth"
    torch.save(bf.state_dict(), name)
    wandb.join()

    # OBSERVE THE WEIGHTS after training
    # for p in bf.parameters():
    #    print(p)
    # EVALUATION RUN

    # DESIRED_REWARD = torch.FloatTensor([200]).to(device)
    # DESIRED_HORIZON = torch.FloatTensor([200]).to(device)
    # desired = DESIRED_REWARD.item()

    # env = gym.make('CartPole-v0')
    # env.reset()
    # rewards = 0
    # while True:
    # command = torch.cat(
    # (DESIRED_REWARD * self.return_scale, DESIRED_HORIZON * self.horizon_scale),
    # dim=-1)
    # # env.render()
    # probs_logits = bf(torch.from_numpy(state).float().to(device), command)
    # probs = torch.softmax(probs_logits, dim=-1).detach().cpu()
    # action = torch.argmax(probs).item()
    # state, reward, done, info = env.step(action)
    # rewards += reward
    # DESIRED_REWARD -= reward
    # DESIRED_HORIZON -= 1
    # if done:
    # break

    # print(
    # "Desired rewards: {} | after finishing one episode the agent received {} rewards"
    # .format(desired, rewards))
    # env.close()


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(main, n_trials=100)
    study.best_params
