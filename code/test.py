from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable

# sources
# https://github.com/Kautenja/gym-super-mario-bros
# https://vmayoral.github.io/robots,/ai,/deep/learning,/rl,/reinforcement/learning/2016/08/07/deep-convolutional-q-learning/

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

################### Model #######################

class CNN(nn.Module):
    def __init__(self, n_channels, n_actions):
        super(CNN, self).__init__()

        # convolutions
        self.conv1 = nn.Conv2d(n_channels, 6, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2, 1)

        self.conv2 = nn.Conv2d(6, 16, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2, 1)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2, 1)

        self.avgpool = nn.AvgPool2d(1, 1)

        # fully connecteds
        self.fc1 = nn.Linear(32 * 7 * 8, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, n_actions)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # convolutions
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.avgpool(x)

        # fully connecteds
        x = x.view(-1, 32 * 7 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def train(env):

    #print(env.action_space.sample())
    # hyperparameters
    epsilon = 0.1

    # initialize model
    model = CNN(3, 12)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # initialize environment and start with random action
    env.reset()
    state, reward, done, info = env.step(env.action_space.sample())

    # loop for a set amount of actions
    for step in range(1000):
        if done:
            state = env.reset()

        # downsample the image
        state = scipy.misc.imresize(state, 20, 'nearest')

        # convert state to tensor and preprocess it for pytorch network
        state = torch.FloatTensor(torch.from_numpy(state).float())
        state = state.permute(2,0,1)
        state = state.view(1, state.shape[0], state.shape[1], state.shape[2])

        # run state through model to get predictions
        out = model(state)
        _, action = out.max(1)

        # convert action to compatible action (tensor to numpy)
        action = action.data.numpy()[0]

        # choose best action with probability 1-e (epsilon-greedy)
        # predict the action given the model
        if np.random.random_sample() > epsilon:
            state, reward, done, info = env.step(action)
        else:
            state, reward, done, info = env.step(env.action_space.sample())  # enter integer between 0 and 11

        # experience replay

        # loop through epochs



        # perform action

        env.render()

    env.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', default='SuperMarioBros-v3', type=str,
                        help='max number of epochs')
    parser.add_argument('-m', default='human', type=str,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    #env = gym_super_mario_bros.make('SuperMarioBrosNoFrameskip-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

    train(env)