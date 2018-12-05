import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm as _tqdm
import random
import time
import torch
import torch.nn as nn
import torch.functional as F
import gym
from CartPoleHelpers import *

class QNetwork(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
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




env = gym.envs.make("CartPole-v0")


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  # keep track of episode duration
    for i in range(num_episodes):

        t = 0
        state = env.reset()

        # Take actions until end of episode
        done = False
        while not done:
            # calculate epsilon for policy
            epsilon = get_epsilon(global_steps)

            # pick action according to q-values and policy
            action = select_action(model, state, epsilon)

            # show demo of the final episode
            if i == num_episodes - 1:
                env.render()
                time.sleep(0.05)

            next_state, reward, done, _ = env.step(action)

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
num_episodes = 100
batch_size = 64
discount_factor = 0.8
learn_rate = 1e-3
memory = ReplayMemory(10000)
num_hidden = 128
seed = 42  # This is not randomly chosen

# We will seed the algorithm (before initializing QNetwork!) for reproducability
random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(0)

model = QNetwork(num_hidden)

episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)

