import gym
import minerl
import logging
# logging.basicConfig(level=logging.DEBUG)

#env = gym.make('MineRLBasaltFindCave-v0')
env = gym.make("MineRLTreechop-custom")  # A MineRLTreechop-v0 env
obs = env.reset()
#env.render()
done = False
slow = 0
while not done:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    env.render()

env.close()