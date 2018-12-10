import os
from tqdm import trange
import time
import torch.nn as nn
import torch.functional as F
from torch import optim
import gym
from helpers import *
import copy
import argparse
from tensorboardX import SummaryWriter
import pickle

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
            epsilon = get_epsilon(global_steps, args.final_epsilon, args.epsilon_threshold)

            # pick action according to q-values and policy
            action = select_action(model, state, epsilon)

            next_state, reward, done, _ = env.step(action)

            # Keep track of highest reached position
            if state[0] > max_position:
                max_position = state[0]

            current_episode.append(action) #(state, action, reward, next_state, done))

            # only sample if there is enough memory
            if len(memory) > batch_size:

                if args.target != 'off':
                    if count % args.update_target == 0:
                        frozen_model = update_target(model, frozen_model, args.target, args.tau)
                    loss = train_target(model, frozen_model, memory, optimizer, batch_size, discount_factor)
                else:
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

        episode_durations.append(t)
        distances.append(max_position)

    #print(episode_durations)
    print("Average episode duration: ", sum(episode_durations)/len(episode_durations))

    #env.reset()
    #for act in best_episode:
    #    env.step(act)
    #    env.render()
    #    time.sleep(0.05)

    print("Max reach")
    print(max_position)
    env.close()
    return distances, episode_durations

def main():

    # We will seed the algorithm (before initializing QNetwork!) for reproducability
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # env.seed(args.seed)
    # np.random.seed(0)


    for i in range(args.num_runs):
        memory = ReplayMemory(args.memory_size, args.sampling)

        model = QNetwork(args.num_hidden)

        distances, episode_durations = run_episodes(train, model, memory, env, args.num_episodes,
                                                    args.batch_size, args.discount_factor, args.learn_rate)

        path = 'MC_' + args.target + '_' + args.sampling + '/'

        if not os.path.exists(path):
            os.makedirs(path)

        filename = 'maxdistance_run:' + str(i)

        with open(path+filename, 'wb') as handle:
            pickle.dump(distances, handle)

        filename = 'steps_run:' + str(i)

        with open(path+filename, 'wb') as handle:
            pickle.dump(episode_durations, handle)

    filename = 'final_model'
    torch.save(model, path + filename + '.pt')

# Arguments and device
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--num_runs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--discount_factor', type=float, default=0.97)
    parser.add_argument('--learn_rate', type=float, default=5e-4)
    parser.add_argument('--num_hidden', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--update_target', type=int, default=100)
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument('--final_epsilon', type=float, default=0.05)
    parser.add_argument('--target', type=str, default='off')
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--epsilon_threshold', type=int, default=1000)
    parser.add_argument('--sampling', type=str, default='prioritized', help='Experience sampling: "off", "prioritized", or "uniform".')
    parser.add_argument('--name', type=str, default='', help='name for tensorboardX run file')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tensorboardX writer
    runs_dir = 'runs/' + args.name
    writer = SummaryWriter(runs_dir)

    main()
