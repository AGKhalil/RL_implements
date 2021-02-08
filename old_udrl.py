import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import gym
import optuna
import argparse


# Behavior Function
class BF(nn.Module):
    def __init__(self, args):
        super(BF, self).__init__()
        self.return_scale = args.return_scale
        self.horizon_scale = args.horizon_scale
        torch.manual_seed(args.seed)
        self.actions = np.arange(args.action_space)
        self.fc1 = nn.Linear(args.state_space, args.hidden_size)
        self.commands = nn.Linear(2, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.action_space)
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


# Replay rBuffer
class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.rbuffer = []

    def add_sample(self, states, actions, rewards):
        episode = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "summed_rewards": sum(rewards)
        }
        self.rbuffer.append(episode)

    def sort(self):
        # sort rbuffer
        self.rbuffer = sorted(self.rbuffer,
                              key=lambda i: i["summed_rewards"],
                              reverse=True)
        # keep the max rbuffer size
        self.rbuffer = self.rbuffer[:self.max_size]

    def get_random_samples(self, batch_size):
        self.sort()
        idxs = np.random.randint(0, len(self.rbuffer), batch_size)
        batch = [self.rbuffer[idx] for idx in idxs]
        return batch

    def get_nbest(self, n):
        self.sort()
        return self.rbuffer[:n]

    def __len__(self):
        return len(self.rbuffer)


class UDRL():
    def __init__(self, args, env, rbuffer, bf, optimizer, device):
        self.last_few = args.last_few
        self.rbuffer = rbuffer
        self.state_space = args.state_space
        self.env = env
        self.action_space = args.action_space
        self.max_reward = args.max_reward
        self.totalnum_iterations = args.totalnum_iterations
        self.device = device
        self.replay_size = args.replay_size
        self.n_warm_up_episodes = args.n_warm_up_episodes
        self.n_updates_per_iter = args.n_updates_per_iter
        self.n_episodes_per_iter = args.n_episodes_per_iter
        self.batch_size = args.batch_size
        self.init_desired_reward = args.init_desired_reward
        self.init_time_horizon = args.init_time_horizon
        self.rbuffer = rbuffer
        self.bf = bf
        self.optimizer = optimizer
        self.return_scale = args.return_scale
        self.horizon_scale = args.horizon_scale

    def sampling_exploration(self):
        """
        This function calculates the new desired reward and new desired horizon
        based on the replay rbuffer. New desired horizon is calculted by the
        mean length of the best last X episodes. New desired reward is sampled
        from a uniform distribution given the mean and the std calculated from
        the last best X performances. where X is the hyperparameter last_few.
        """

        top_X = self.rbuffer.get_nbest(self.last_few)
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
        Given a saved episode from the replay rbuffer this function samples
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
        rbuffer episodes are build like [cumulative episode reward, states,
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
        # select randomly episodes from the rbuffer
        episodes = self.rbuffer.get_random_samples(batch_size)
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
        the actions appropriate to the states from the replay rbuffer.
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
        Generates more samples for the replay rbuffer.
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
    def run_upside_down(self, wandb, totalnum_iterations):
        """
        """
        all_rewards = []
        losses = []
        average_100_reward = []
        desired_rewards_history = []
        horizon_history = []
        for training_step in range(1, totalnum_iterations + 1):

            # improve|optimize bf based on replay rbuffer
            loss_buffer = []
            for i in range(self.n_updates_per_iter):
                bf_loss = self.train_behavior_function(self.batch_size)
                loss_buffer.append(bf_loss)
            bf_loss = np.mean(loss_buffer)
            losses.append(bf_loss)

            # run x new episode and add to rbuffer
            for i in range(self.n_episodes_per_iter):

                # Sample exploratory commands based on rbuffer
                new_desired_reward, new_desired_horizon = self.sampling_exploration(
                )
                generated_episode = self.generate_episode(
                    new_desired_reward, new_desired_horizon)
                self.rbuffer.add_sample(generated_episode[0],
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
                step=training_step)

            print(
                "\rEpisode: {} | Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f}"
                .format(training_step, ep_rewards, np.mean(all_rewards[-100:]),
                        bf_loss),
                end="",
                flush=True)
            if training_step % 100 == 0:
                print(
                    "\rEpisode: {} | Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f}"
                    .format(training_step, ep_rewards,
                            np.mean(all_rewards[-100:]), bf_loss))

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

            self.rbuffer.add_sample(states, actions, rewards)


def optuna_study(trial):
    wandb.init(entity="agkhalil", project="pytorch-udrl-cartpole", reinit=True)
    wandb.watch_called = False
    config = wandb.config
    env = gym.make("CartPole-v0")
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    config.max_reward = 200
    config.totalnum_iterations = 200
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

    # init replay rbuffer with n-warmup runs
    rbuffer = ReplayBuffer(config.replay_size)
    bf = BF(state_space, action_space, 64, 1, config.horizon_scale,
            config.return_scale).to(config.device)
    optimizer = optim.Adam(params=bf.parameters(), lr=1e-3)
    udrl = UDRL(env, action_space, state_space, config.max_reward,
                config.totalnum_iterations, config.device, config.replay_size,
                config.n_warm_up_episodes, config.n_updates_per_iter,
                config.n_episodes_per_iter, config.last_few, config.batch_size,
                config.init_desired_reward, config.init_time_horizon, rbuffer,
                bf, optimizer, config.return_scale, config.horizon_scale)
    udrl.warmup()
    rewards, average, d, h, loss = udrl.run_upside_down(
        wandb, totalnum_iterations=config.max_episodes)

    # SAVE MODEL
    name = "model.pth"
    torch.save(bf.state_dict(), name)
    wandb.join()


def main(args):
    wandb.init(entity="agkhalil", project="udrl-contrast")
    env = gym.make(args.env_name)
    device = torch.device(args.device)
    rbuffer = ReplayBuffer(args.replay_size)
    bf = BF(args).to(device)
    optimizer = optim.Adam(params=bf.parameters(), lr=args.lr)
    udrl = UDRL(args, env, rbuffer, bf, optimizer, device)
    udrl.warmup()
    rewards, average, d, h, loss = udrl.run_upside_down(
        wandb, args.totalnum_iterations)

    torch.save(bf.state_dict(), wandb.run.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='optimizer learning rate')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=60,
                        help='BF hidden size')
    parser.add_argument('--seed', type=int, default=42, help='Experiment seed')
    parser.add_argument('--optuna',
                        type=bool,
                        default=False,
                        help='Optuna study running')
    parser.add_argument('--env_name',
                        type=str,
                        default='CartPole-v0',
                        help='Gym env')
    args, remaining_args = parser.parse_known_args()
    env = gym.make(args.env_name)
    parser.add_argument('--state_space',
                        type=int,
                        default=env.observation_space.shape[0],
                        help='env state space')
    parser.add_argument('--action_space',
                        type=int,
                        default=env.action_space.n,
                        help='env action space')
    parser.add_argument('--max_reward',
                        type=int,
                        default=env.spec.reward_threshold,
                        help='Max reward allowed')
    parser.add_argument('--totalnum_iterations',
                        type=int,
                        default=100,
                        help='Total number of episodes to run')
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:1" if torch.cuda.is_available() else "cpu",
        help='cuda:1 if available else cpu')
    parser.add_argument('--horizon_scale',
                        type=float,
                        default=0.02,
                        help='Desired horizon scale')
    parser.add_argument('--return_scale',
                        type=float,
                        default=0.03,
                        help='Desired return scale')
    parser.add_argument('--replay_size',
                        type=int,
                        default=700,
                        help='Replay rbuffer size')
    parser.add_argument('--n_warm_up_episodes',
                        type=int,
                        default=30,
                        help='number of warmup episodes')
    parser.add_argument('--init_time_horizon',
                        type=int,
                        default=1,
                        help='initial desired time horizon')
    parser.add_argument('--init_desired_reward',
                        type=int,
                        default=1,
                        help='initial desired reward')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1536,
                        help='batch size')
    parser.add_argument(
        '--last_few',
        type=int,
        default=25,
        help='Number of high ranking episodes to use to sample commands')
    parser.add_argument('--n_episodes_per_iter',
                        type=int,
                        default=40,
                        help='number of new episodes collected per iteration')
    parser.add_argument('--n_updates_per_iter',
                        type=int,
                        default=100,
                        help='number of policy updates per iteration')

    args = parser.parse_args()

    if args.optuna:
        study = optuna.create_study()
        study.optimize(optuna_study, n_trials=100)
        study.best_params
    else:
        main(args)
