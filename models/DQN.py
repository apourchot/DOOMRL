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
learning_rate = 0.00025
beta_1 = 0.95
beta_2 = 0.999
replay_memory_size = 20000
channels = 3
resolution = (32, 32)
state_shape = (replay_memory_size, channels, resolution[0], resolution[1])
batch_size = 64
frame_repeat = 12


def preprocess(img):
    img = skimage.transform.resize(img, resolution, mode='constant')
    img = img.astype(np.float32)
    img = np.moveaxis(img, 2, 0)
    img = img.reshape((1, channels, resolution[0], resolution[1]))
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

        self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not is_terminal:
            self.s2[self.pos, :, :, :] = s2
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
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32,  32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32,  64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64,  64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, nb_available_actions)

    def forward(self, x):
        LReLU = F.leaky_relu
        x = LReLU(self.conv1(x))
        # x = LReLU(self.conv2(x))
        x = self.pool(x)
        x = LReLU(self.conv3(x))
        # x = LReLU(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 256)
        x = LReLU(self.fc1(x))
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
                 loading):
        self.game = game_instance
        self.actions = actions
        self.state_shape = state_shape
        self.file_name = file_name
        self.memory = ReplayMemory(capacity=replay_memory_size)
        if(loading):
            self.net = torch.load(file_name)
        else:
            self.net = Net(len(actions))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate,
                                            betas=(beta_1, beta_2))
        self.n_epoch = 1

    def q_values(self, s):

        s = torch.from_numpy(s)
        s = Variable(s, requires_grad=False)
        return self.net(s)

    def exploration_rate(self, n_epoch):

        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 1000 * n_epoch  # 10% of learning time
        eps_decay_epochs = 6000 * n_epoch  # 60% of learning time

        if n_epoch < const_eps_epochs:
            return start_eps
        elif n_epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    def step(self, training):

        if(training):
            s1 = preprocess(self.game.get_state().screen_buffer)
            # Choosing action according to epsilon greedy policy #
            epsilon = self.exploration_rate(self.n_epoch)
            if random() <= epsilon:
                a = randint(0, len(self.actions) - 1)
            else:
                _, index = torch.max(self.q_values(s1), 1)
                a = index.data.numpy()[0]

            reward = self.game.make_action(self.actions[a], frame_repeat)
            is_terminal = self.game.is_episode_finished()
            if(not is_terminal):
                s2 = preprocess(self.game.get_state().screen_buffer)
            else:
                s2 = None

            # Optimization #
            self.memory.push(s1, a, s2, is_terminal, reward)
            if self.memory.size > batch_size:
                s1, a, s2, is_terminal, r = self.memory.get_sample(batch_size)
                q_max = np.max(self.q_values(s2).data.numpy(), axis=1)
                target_q = self.q_values(s1).data.numpy()
                delta_q = r + discount_factor * (1 - is_terminal) * q_max
                target_q[np.arange(target_q.shape[0]), a] = delta_q
                self.net.learn(s1, target_q, self.optimizer)
                torch.save(self.net, self.file_name)

        else:
            s1 = preprocess(self.game.get_state().screen_buffer)

            # Choosing best action #
            _, index = torch.max(self.q_values(s1), 1)
            a = index.data.numpy()[0]

            reward = self.game.set_action(self.actions[a])
            for _ in range(frame_repeat):
                self.game.advance_action()


        self.n_epoch += 1

        return a
