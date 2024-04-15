import warnings
import time
from datetime import datetime

import gym
from termcolor import colored
import minerl
import helperThings

from stable_baselines3.common import evaluation
from tqdm.notebook import tqdm
import stable_baselines3
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack , DummyVecEnv
from stable_baselines3 import PPO, DQN

def createEnv():
    minerl_env = gym.make(envName)
    minerl_env.seed(seed)

    ###### WRAP ENVIRONMENT ######
    obs_wrapped_diamond = helperThings.ExtractPOV(minerl_env)
    print(colored("*****EXTRACTED POV*****", 'red'))
    obs_action_wrapped_diamond = helperThings.ActionShaping(obs_wrapped_diamond)
    print(colored("*****ACTIONS SHAPED*****", 'red'))

    ###### FRAME STACKING ######
    wrapped_env = VecFrameStack(DummyVecEnv([lambda: obs_action_wrapped_diamond]), n_stack=4)

    # measure how long it takes for an env reset
    res1 = time.perf_counter()
    wrapped_env.reset()
    end1 = time.perf_counter()

    print(colored("Time spent on reset: " + str(end1 - res1), 'magenta'))

    return wrapped_env


warnings.simplefilter('ignore')
seed = 21
envName = "MineRLObtainDiamondShovel-v0"
totalTimesteps = 10000
logInterval = 500
#envID = 0
#iterations = 5
#show = 1
log_name = "run"+datetime.now().strftime("%H:%M:%S")
# seed = input("Seed: ")
show = int(input("\nPlay model after training: (1 = yes, 0 = no)\n"))
iterations = int(input("\nHow many iterations: \n"))
envID = int(input("\nSelect ENV:\n0:MineRLObtainDiamondShovel-v0\n1:MineRLTreechop-v0\n2:MineRLTreechop-custom\n"))
if envID == 0:
    envName = "MineRLObtainDiamondShovel-v0"
elif envID == 1:
    envName = "MineRLTreechop-v0"
elif envID == 2:
    envName = "MineRLTreechop-custom"
else:
    envName = "MineRLObtainDiamondShovel-v0"
# totalTimestamps = input("Total timesteps: ")
# logInterval = input("Log interval: ")


start_time = time.perf_counter()
wrapped_env = createEnv()
print(colored("*****CREATING AND TRAINING MODEL*****", 'yellow'))
model = PPO(policy="CnnPolicy", env=wrapped_env, verbose=1, seed = seed, tensorboard_log="./L", learning_rate=0.003, n_steps=500)

for i in range(iterations):
    wrapped_env.render()
    iterr = time.perf_counter()
    print(colored("iteration " + str(i+1) + " starting", 'green'))
    model.learn(total_timesteps=totalTimesteps, reset_num_timesteps=False, tb_log_name=log_name + "_" + str(i))
    wrapped_env.reset()
    enditer = time.perf_counter()
    print(colored("Time spent on iteration " + str(i+1) + " : " + str(enditer - iterr), 'magenta'))

model.save("./models")
print(colored("*****MODEL CREATED AND SAVED*****", 'red'))


print(colored("*****RESETTING ENV*****", 'yellow'))
obs = wrapped_env.reset()
print(colored("*****ENV RESET*****", 'red'))

end_time = time.perf_counter()
print("Elapsed time for training and resetting: ", end_time - start_time)

if show == 1:
    # parameters for visualising the model
    done = False
    total_reward = 0
    steps_taken = 0
    max_steps = 8000


    print(colored("*****STARTING RENDER AND WHILE LOOP*****", 'yellow'))

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = wrapped_env.step(action)
        print(str(total_reward) + " " + str(reward)) if (steps_taken % 100) == 0 else None
        if total_reward < total_reward + reward:
            print(colored(total_reward+reward, 'red') + " at timestep: " + str(steps_taken))
        total_reward += reward
        steps_taken += 1
        if max_steps is not None and steps_taken > max_steps:
            break
        if done:
            break
        wrapped_env.render()
    print("Total rewards after test run: " + str(total_reward))