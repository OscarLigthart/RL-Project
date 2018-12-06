import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm as _tqdm
from tqdm import trange
import random
import time
import torch
import torch.nn as nn
import torch.functional as F
import gym
from helpers import *

class QNetwork(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(2, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]

        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


env = gym.envs.make("MountainCar-v0")


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  # keep track of episode duration
    for i in trange(num_episodes):

        t = 0
        state = env.reset()

        # Take actions until end of episode
        done = False

        max_position = -.4
        add_reward = 0
        successful = []

        while not done:
            # calculate epsilon for policy
            epsilon = get_epsilon(global_steps, final_epsilon, flatline)

            # pick action according to q-values and policy
            action = select_action(model, state, epsilon)

            # show demo of the final episode
            if i == num_episodes - 1:
                env.render()
                time.sleep(0.05)

            next_state, reward, done, _ = env.step(action)

            # # Give a reward for reaching a new maximum position
            # if state[0] > max_position:
            #     max_position = state[0]
            #     add_reward += 10
            # else:
            #     add_reward += reward


            memory.push((state, action, reward, next_state, done))

            # only sample if there is enough memory
            if len(memory) > batch_size:
                loss = train(model, memory, optimizer, batch_size, discount_factor)

            state = next_state
            global_steps += 1
            t += 1

        episode_durations.append(t)

    print(episode_durations)
    env.close()
    return episode_durations

# Let's run it!
num_episodes = 10000
batch_size = 64
discount_factor = 0.97
learn_rate = 5e-4
memory = ReplayMemory(10000)
num_hidden = 20 #128
seed = 42  # This is not randomly chosen

# Epsilon function linearly decreases until certain number of iterations, after which it is constant
final_epsilon = 0.05
flatline = 5000 # Turning point linear -> constant

# We will seed the algorithm (before initializing QNetwork!) for reproducability
random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(0)

model = QNetwork(num_hidden)

episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
