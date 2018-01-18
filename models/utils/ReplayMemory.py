from random import sample, randint, random, randint, uniform
from collections import namedtuple
from time import time
import numpy as np
import torch
from torch.autograd import Variable
from .SumTree import SumTree

# pytorch tensors
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

# Transitions
Transition = namedtuple('Transition', ('state', 'variable', 'action', 'next_state', 'next_variable', 'reward'))
TransitionV2 = namedtuple('TransitionV2', ('state', 'variable', 'action', 'next_state', 'next_variable', 'reward', 'index', 'weight'))

# dqn parameters - not ok to change will affect test
RESOLUTION = (60, 108)
NB_CHANNELS = 3

# dqn parameters - ok to change won't affect test
DATA_AUGMENTATION = False

class ReplayMemory(object):

    def __init__(self, capacity, nb_game_variables, prioritized=False):

        self.capacity = capacity
        self.prioritized = prioritized
        self.states = torch.zeros(capacity, 1, NB_CHANNELS, RESOLUTION[0], RESOLUTION[1]).type(FloatTensor)
        self.variables = torch.zeros(capacity, 1, nb_game_variables)
        self.rewards = torch.zeros(capacity, 1).type(FloatTensor)
        self.actions = torch.zeros(capacity, 1, 1).type(LongTensor)
        self.is_terminal = torch.zeros(capacity).type(ByteTensor)
        self.size = 0
        if(prioritized):
            self.sum_tree = SumTree(capacity)
        self.position = 0

    # add a transition in memory
    def push(self, *args):
        T = Transition(*args)
        self.size = min(self.capacity, self.size + 1)
        self.variables[self.position] = T.variable
        self.states[self.position] = T.state
        self.actions[self.position] = T.action
        self.rewards[self.position] = T.reward
        self.is_terminal[self.position] = T.next_state is None
        if(self.prioritized):
            if(self.size == 1):
                weight = 1
            else:
                weight = self.sum_tree.max()
            self.sum_tree.add(weight)
        self.position = (self.position + 1) % self.capacity

    # updating prior for prioritized replay
    def update_prior(self, indexes, priors):
        a = time()
        for i in range(len(indexes)):
            self.sum_tree.set(indexes[i], priors[i])
        self.sum_tree.update_tree()

    # sample a batch from memory
    def sample(self, batch_size):
        samples = []
        if(self.prioritized):
            segment = self.sum_tree.total() / batch_size
            total = self.sum_tree.total()
        for i in range(batch_size):

            # picking random element from memory
            if(self.prioritized):
                u = uniform(segment * i, segment * (i + 1))
                index, p = self.sum_tree.get(u)
                weigth = FloatTensor([1 / (p/total * self.size)])
            else:
                index = randint(0, self.size - 1)
            state = self.states[index]
            next_state = self.states[(index + 1) % self.capacity] if not self.is_terminal[index] else None
            variable = self.variables[index]
            next_variable = self.variables[(index + 1) % self.capacity] if not self.is_terminal[index] else None
            action = self.actions[index]
            reward = self.rewards[index]

            # adding to samples
            if(self.prioritized):
                samples += [TransitionV2(state, variable, action, next_state, next_variable, reward, LongTensor([index]), weigth)]
            else:
                samples += [Transition(state, variable, action, next_state, next_variable, reward)]
        if(self.prioritized):
            batch = TransitionV2(*zip(*samples))
        else:
            batch = Transition(*zip(*samples))

        return batch

    # sample batch_size short episodes of length episode_length from memory
    def sample_episode(self, batch_size, episode_length):

        a = time()

        # misc
        i = 0
        episodes = []
        cpt = 0
        if(self.prioritized):
            segment = self.sum_tree.total() / self.size

        while(i < batch_size):

            # sampling from memory
            if(self.prioritized):
                u = uniform(segment * i, segment * (i + 1))
                begin, p = self.sum_tree.get(u)
                weigth = FloatTensor([segment / p])
            else:
                begin = randint(0, self.size - 1)

            # is_terminal, action and reward value for complete sequence
            is_terminal = self.is_terminal[begin]
            action = self.actions[begin]
            reward = self.rewards[begin]

            # creating history for this sample
            frames = self.states[begin % self.capacity].view(1, 1,
            NB_CHANNELS, RESOLUTION[0], RESOLUTION[1])
            frames_next = self.states[(begin + 1) % self.capacity].view(1, 1,
                                NB_CHANNELS, RESOLUTION[0], RESOLUTION[1]) if not is_terminal else None

            # populating history
            for j in range(1, episode_length):

                # did we jump from one episode to another ?
                ep_jump = False # self.is_terminal[(begin - j) % self.capacity]

                # if we jumped then we can't go further
                if(ep_jump):
                    break

                # adding frames to history
                curr_frame = self.states[(begin - j) % self.capacity].view(1, 1,
                NB_CHANNELS, RESOLUTION[0], RESOLUTION[1])
                frames = torch.cat((curr_frame, frames), 0)
                if(not is_terminal):
                    next_frame = self.states[(begin - j + 1) % self.capacity].view(1, 1,
                                        NB_CHANNELS, RESOLUTION[0], RESOLUTION[1])
                    frames_next = torch.cat((next_frame, frames_next), 0)

            # adding to samples
            if(self.prioritized):
                episodes += [TransitionV2(frames, action, frames_next, reward, LongTensor([begin]), weigth)]
            else:
                episodes += [Transition(frames, action, frames_next, reward)]
            i += 1

        if(self.prioritized):
            batch = TransitionV2(*zip(*episodes))
        else:
            batch = Transition(*zip(*episodes))

        return batch
