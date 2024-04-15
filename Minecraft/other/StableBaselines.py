import gym
import minerl
from logging import getLogger
import numpy as np
import argparse

from termcolor import colored

import wrapper
from stable_baselines.common.policies import CnnPolicy, MlpPolicy
from stable_baselines import PPO1, PPO2
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
import os
import sys

from helperThings import ExtractPOV, ActionShaping

# import logging
# logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='MineRLTreechop-v0')
parser.add_argument('--log_dir', type=str, default='home/bagpla/Desktop/mineRL/logs')
#parser.add_argument('--n_cpu', type=int, default=10)
parser.add_argument('--n_timesteps', type=int, default=1000)
parser.add_argument('--save_dir', type=str, default='home/bagpla/Desktop/mineRL/models')
args = parser.parse_args()


#env = make_vec_env(env_id=args.env, n_envs=args.n_cpu, seed=23)
env = gym.make("MineRLBasaltFindCave-v0")
obs_wrapped = ExtractPOV(env)
print(colored("*****EXTRACTED POV*****", 'blue'))
obs_actions = ActionShaping(obs_wrapped)
print(colored("*****ACTIONS SHAPED*****", 'blue'))
obs = obs_actions.reset()
print(colored("*****ENV RESET*****", 'blue'))


#model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=args.log_dir).learn(total_timesteps=args.n_timesteps)
model = PPO2(CnnPolicy, obs_actions, verbose=1, tensorboard_log=args.log_dir).learn(total_timesteps=args.n_timesteps)

model.save(args.save_dir)
print(colored("*****MODEL CREATED AND SAVED*****", 'blue'))
#eval_env = gym.make(args.env)

#mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
#print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

total_reward = 0
obs_actions.reset()
print(colored("*****ENTERING WHILE LOOP*****", 'yellow'))
done = False
while not done:
    t = 0
    action, _states = model.predict(obs_actions)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    print(total_reward)
    env.render()
    t += 1
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

env.close()