import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, A2C
import wandb


# Parallel environments
env = make_vec_env('MountainCarContinuous-v0', n_envs=1)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=260000)

ep_reward = 0
steps= 0
done = False
obs = env.reset()
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    ep_reward += rewards
    steps += 1
print(ep_reward, steps)
