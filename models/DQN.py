from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage.transform
import skimage.color
from torchvision import datasets, transforms
from torch.autograd import Variable
import skimage
from tqdm import trange

# hyperparameters
discount_factor = 0.99
learning_rate = 0.00001
replay_memory_size = 1000
channels = 1
resolution = (30, 45)
state_shape = (replay_memory_size, channels, resolution[0], resolution[1])
batch_size = 64
frame_repeat = 12
epsilon = 0.6


def preprocess(img):
    img = skimage.transform.resize(img, resolution, mode='constant')
    img = img.astype(np.float32)
    return img


class ReplayMemory:

    def __init__(self, capacity):

        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.is_terminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def push(self, s1, action, s2, is_terminal, reward):

        self.s1[self.pos, 0, :, :] = s1
        self.a[self.pos] = action
        if not is_terminal:
            self.s2[self.pos, 0, :, :] = s2
        self.is_terminal[self.pos] = is_terminal
        self.r[self.pos] = reward
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, batch_size):

        i = sample(range(0, self.size), batch_size)
        return (self.s1[i], self.a[i], self.s2[i],
                self.is_terminal[i], self.r[i])


class Net(nn.Module):

    def __init__(self, nb_available_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, nb_available_actions)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def learn(self, s1, target_q, optimizer):

        # computing loss
        s1 = Variable(torch.from_numpy(s1))
        target_q = Variable(torch.from_numpy(target_q), requires_grad=False)
        curr_q = self.forward(s1)
        criterion = nn.SmoothL1Loss()
        loss = criterion(curr_q, target_q)

        # gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


class DQN():

    def __init__(self, game_instance, actions, file_name,
                 loading, training):
        self.game = game_instance
        self.actions = actions
        self.state_shape = state_shape
        self.file_name = file_name
        self.memory = ReplayMemory(capacity=replay_memory_size)
        if(loading):
            self.net = torch.load(file_name)
        else:
            self.net = Net(len(actions))
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=learning_rate)
        self.training = training

    def q_values(self, s):

        s = torch.from_numpy(s)
        s = Variable(s, requires_grad=False)
        return self.net(s)

    def step(self):

        s1 = preprocess(self.game.get_state().screen_buffer)

        # Choosing action and going to next state #
        eps = epsilon
        if random() <= eps:
            a = randint(0, len(self.actions) - 1)
        else:
            s1 = s1.reshape([1, channels, resolution[0], resolution[1]])
            _, index = torch.max(self.q_values(s1), 1)
            a = index.data.numpy()[0]

        reward = self.game.set_action(self.actions[a])
        for _ in range(frame_repeat):
            self.game.advance_action()
        is_terminal = self.game.is_episode_finished()
        if(not is_terminal):
            s2 = preprocess(self.game.get_state().screen_buffer)
        else:
            s2 = None

        # Optimization #
        if(self.training):
            self.memory.push(s1, a, s2, is_terminal, reward)
            if self.memory.size > batch_size:
                s1, a, s2, is_terminal, r = self.memory.get_sample(batch_size)
                q_max = np.max(self.q_values(s2).data.numpy(), axis=1)
                target_q = self.q_values(s1).data.numpy()
                delta_q = r + discount_factor * (1 - is_terminal) * q_max
                target_q[np.arange(target_q.shape[0]), a] = delta_q
                self.net.learn(s1, target_q, self.optimizer)
                torch.save(self.net, self.file_name)

        return a
