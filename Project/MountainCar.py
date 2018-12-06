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
import copy

class QNetwork(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(2, num_hidden)
        self.l2 = nn.Linear(num_hidden, 3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


env = gym.envs.make("MountainCar-v0")


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  # keep track of episode duration

    count = 0
    frozen_model = copy.deepcopy(model)

    best_episode = []
    distances = []

    max_position = -.4

    for i in trange(num_episodes):

        t = 0
        state = env.reset()

        # Take actions until end of episode
        done = False

        current_episode = []
        before = max_position

        # reset max position
        max_position = -.4

        while not done:
            # calculate epsilon for policy
            epsilon = get_epsilon(global_steps, final_epsilon, flatline)

            # pick action according to q-values and policy
            action = select_action(model, state, epsilon)

            # show demo of the final episode

            next_state, reward, done, _ = env.step(action)

            # # Give a reward for reaching a new maximum position
            if state[0] > max_position:
                max_position = state[0]
            #     add_reward += 10
            # else:
            #     add_reward += reward

            current_episode.append(action) #(state, action, reward, next_state, done))

            # only sample if there is enough memory
            if len(memory) > batch_size:
                #
                # if target:
                    # if count % update_target == 0:
                    #     frozen_model = copy.deepcopy(model)
                #     loss = train_target(model, frozen_model, memory, optimizer, batch_size, discount_factor)
                # else:
                #     loss = train(model, memory, optimizer, batch_size, discount_factor)

                loss = train(model, memory, optimizer, batch_size, discount_factor)
                memory.push((state, action, reward, next_state, done, loss))

            else:
                memory.push((state, action, reward, next_state, done, 0))

            state = next_state
            global_steps += 1
            t += 1
            count += 1

        if max_position > before:
            best_episode = current_episode

        #print(max_position)
        episode_durations.append(t)
        distances.append(max_position)

    print(episode_durations)
    print("Average episode duration: ", sum(episode_durations)/len(episode_durations))

    env.reset()
    for act in best_episode:
        env.step(act)
        env.render()
        time.sleep(0.05)

    print("Max reach")
    print(max_position)
    env.close()
    return distances

# Let's run it!
num_episodes = 500
batch_size = 64
discount_factor = 0.97
learn_rate = 5e-4
memory = ReplayMemory(10000, 'prioritized')
num_hidden = 200 #128
seed = 42  # This is not randomly chosen
target = True
update_target = 100

# Epsilon function linearly decreases until certain number of iterations, after which it is constant
final_epsilon = 0.05
flatline = 1000 # Turning point linear -> constant

# We will seed the algorithm (before initializing QNetwork!) for reproducability
random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(0)

model = QNetwork(num_hidden)

distances = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)

plt.figure()
plt.plot(distances)
plt.show()
