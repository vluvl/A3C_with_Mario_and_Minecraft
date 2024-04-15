import gym
#from torchvision import transforms as T
#import gym_super_mario_bros
import minerl
import helperThings
import matplotlib.pyplot as plt
from gym.spaces import Box
from gym import Wrapper
#from nes_py.wrappers import JoypadSpace
#from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp
#import torch

class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())

def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        return state, reward , done, info

    def reset(self):
        state = super().reset()
        return process_frame(state)


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                #self.env.render()           #render every step before concatenation
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), total_reward, done, info

    def reset(self):
        state = super().reset()
        self.vertical_angle = 0
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)

def create_train_env_mine(envName, output_path=None):
    env = gym.make(envName)


    env = helperThings.ExtractPOV(env)      #get POV in the right format

    env = helperThings.ActionShaping(env)   #get actions in the right format


    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = CustomReward(env, monitor)        #frame processing
    env = CustomSkipFrame(env)
    env.seed(seed=42)                            #Keep things consistent

    return env, env.observation_space.shape[0], 8
