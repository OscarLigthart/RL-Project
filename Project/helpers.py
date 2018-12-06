import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import random
import copy

class ReplayMemory:

    def __init__(self, capacity, method):
        self.capacity = capacity
        self.memory = []
        self.method = method

    def push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]

        self.memory.append(transition)

    def sample(self, batch_size):
        if self.method == 'uniform':
            return random.sample(self.memory, batch_size)

        elif self.method == 'off':
            return self.memory[-batch_size:]

        elif self.method == 'prioritized':
            # sort on loss
            sort_memory = sorted(self.memory, key=lambda x: x[5])

            # return batch_size amount of samples
            return sort_memory[-batch_size:]

        else:
            raise NotImplementedError('Not a valid method of experience replay, choose between:'
                                      ' off, uniform and prioritized')


    def __len__(self):
        return len(self.memory)

def select_action(model, state, epsilon):
    # calculate the q values
    actions = model(torch.FloatTensor(state))

    # get highest q values
    values, indices = actions.max(0)

    # use policy to choose action
    if random.random() < epsilon:
        #a = np.random.choice(range(len(actions))).item()
        a = random.choice(range(len(actions)))
    else:
        a = indices.item()

    return a

def update_target(model, target_model, soft, tau):
    
    if soft:
        for frozen_parameters, parameters in zip(target_model.parameters(), model.parameters()):
            frozen_parameters.data.copy_(tau*parameters.data + frozen_parameters.data*(1.0 - tau))
    else:
        target_model = copy.deepcopy(model)
        
    return target_model
    

def get_epsilon(it, final_epsilon, flatline):
    # after 1000 iterations, e-greedy with epsilon being 0.5
    if it >= flatline:
        epsilon = final_epsilon
    # linearly decay epsilon before 1000 iterations
    else:
        epsilon = 1 - (1 - final_epsilon) * it * (1 / flatline)

    return epsilon

def compute_q_val(model, state, action):
    # get q-values
    actions = model(state)
    q_val = torch.gather(actions, 1, action.view(-1, 1))

    return q_val


def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)

    # calculate q-values and pick highest
    actions = model(next_state)
    _, chosen_action = actions.max(1)
    chosen_action = torch.gather(actions, 1, chosen_action.view(-1, 1))

    #  calculate target
    target = reward.view(chosen_action.shape) + (discount_factor * chosen_action)

    # set target to just the reward if next_state is terminal
    target[done] = reward[done].view(target[done].shape)

    return target


def train(model, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done, loss = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def train_target(model, target_model, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done, loss = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q valuereturn self.memory[-batch_size:]
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(target_model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())
