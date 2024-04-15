import warnings
import time
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

# import logging
# logging.basicConfig(level=logging.DEBUG)


warnings.simplefilter('ignore')

start_time = time.perf_counter()

###### SET ENVIRONMENT ######
minerl_env = gym.make("MineRLObtainDiamondShovel-v0")
#minerl_env = gym.make("MineRLNavigate-v0")
#minerl_env = gym.make("MineRLTreechop-custom")
minerl_env.seed(21)

obs_wrapped_diamond = helperThings.ExtractPOV(minerl_env)
print(colored("*****EXTRACTED POV*****", 'red'))

obs_action_wrapped_diamond = helperThings.ActionShaping(obs_wrapped_diamond)
print(colored("*****ACTIONS SHAPED*****", 'red'))

##Added support for Frame Stacking
obs_action_wrapped_diamond = DummyVecEnv([lambda: obs_action_wrapped_diamond])
obs_action_wrapped_diamond = VecFrameStack(obs_action_wrapped_diamond, n_stack=4)

res1 = time.perf_counter()
obs_action_wrapped_diamond.reset()
end1 = time.perf_counter()
print(colored("Time spent on first reset: " + str(end1 - res1), 'magenta'))
print(colored("*****CREATING AND TRAINING MODEL*****", 'yellow'))

model = PPO(policy="CnnPolicy", env=obs_action_wrapped_diamond, verbose=1, seed = 21, tensorboard_log="./L", learning_rate=0.1)
#model = DQN(policy="CnnPolicy", env=obs_action_wrapped_diamond, verbose=1, seed = 21, buffer_size= 10000)
for i in range(5):
    iterr = time.perf_counter()
    print(colored("iteration " + str(i+1) + " starting", 'green'))
    model.learn(total_timesteps=500,log_interval=100, reset_num_timesteps=False, tb_log_name="runs")
    print("resetting env...")
    #obs_action_wrapped_diamond.reset()
    enditer = time.perf_counter()
    print(colored("Time spent on iteration " + str(i+1) + " : " + str(enditer - iterr), 'magenta'))
model.save("./models")
print(colored("*****MODEL CREATED AND SAVED*****", 'red'))

# minerl_env.seed(21)
print(colored("*****RESETTING ENV*****", 'yellow'))
obs = obs_action_wrapped_diamond.reset()
print(colored("*****ENV RESET*****", 'red'))

end_time = time.perf_counter()
print("Elapsed time for training and resetting: ", end_time - start_time)

done = False
total_reward = 0
steps_taken = 0
max_steps = None
print(colored("*****STARTING RENDER AND WHILE LOOP*****", 'yellow'))
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = obs_action_wrapped_diamond.step(action)
    print(str(total_reward) + " " + str(reward))
    if total_reward < total_reward + reward:
        print(colored(total_reward+reward, 'red'))
    total_reward += reward
    steps_taken += 1
    if max_steps is not None and steps_taken > max_steps:
        break
    if done:
        break
    obs_action_wrapped_diamond.render()
