from vizdoom import *
import itertools as it
from random import sample, randint, random
from collections import namedtuple
from time import time, sleep
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import skimage.transform
import skimage.color
import skimage.io
import skimage
from tqdm import trange

# hyperparameters
discount_factor = 0.99
learning_rate = 0.00025
beta_1 = 0.95
beta_2 = 0.999
replay_memory_size = 20000
channels = 1
resolution = (32, 32)
state_shape = (replay_memory_size, channels, resolution[0], resolution[1])
batch_size = 64
frame_repeat = 12
eps_start = 1
eps_end = 0.1
eps_decay = 10000


def preprocess(img):
    img = skimage.transform.resize(img, resolution, mode='constant')
    img = img.astype(np.float32)
    img = img.reshape((1, channels, resolution[0], resolution[1]))
    return torch.from_numpy(img).type(torch.FloatTensor)

# preprocess = T.Compose([T.ToPILImage(), T.Resize(64, interpolation=Image.CUBIC), T.ToTensor()])
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):

    def __init__(self, nb_available_actions):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, nb_available_actions)

    def forward(self, x):

        LReLU = F.leaky_relu
        x = LReLU(self.conv1(x))
        x = LReLU(self.conv2(x))
        x = self.pool(x)
        x = LReLU(self.conv3(x))
        x = LReLU(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = LReLU(self.fc1(x))
        return self.fc2(x)


class DQN():

    def __init__(self, game_instance, actions, file_name, loading):

        self.game = game_instance
        self.actions = actions
        self.state_shape = state_shape
        self.file_name = file_name
        self.memory = ReplayMemory(capacity=replay_memory_size)
        if(loading):
            self.net = torch.load(file_name)
        else:
            self.net = Net(len(actions))
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.n_steps = 1

    def learn(self, q_values, expected_q_values):

        # computing loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values)

        # gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.net.parameters(), 1)
        self.optimizer.step()

        return loss

    def select_action(self, n_steps, state, training):

        u = random()
        eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * n_steps / eps_decay)
        if (u > eps_threshold or not training):
            return self.net(Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[randint(0, len(self.actions) - 1)]])

    def step(self, training):

        s1 = preprocess(self.game.get_state().screen_buffer)
        a = self.select_action(self.n_steps, s1, training)

        if(training):

            reward = torch.FloatTensor([self.game.make_action(self.actions[a.numpy()[0][0]], frame_repeat)])
            is_terminal = self.game.is_episode_finished()
            if(not is_terminal):
                s2 = preprocess(self.game.get_state().screen_buffer)
            else:
                s2 = None
            self.memory.push(s1, a, s2, reward)


            if len(self.memory) > batch_size:

                transitions = self.memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
                non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

                state_batch = Variable(torch.cat(batch.state))
                action_batch = Variable(torch.cat(batch.action))
                reward_batch = Variable(torch.cat(batch.reward))

                q_values = self.net(state_batch).gather(1, action_batch)

                next_v_values = Variable(torch.zeros(batch_size).type(torch.FloatTensor))
                next_v_values[non_final_mask] = self.net(non_final_next_states).max(1)[0]

                next_v_values.volatile = False
                expected_q_values = (next_v_values * discount_factor) + reward_batch
                self.learn(q_values, expected_q_values)


        else:
            self.game.set_action(self.actions[a.numpy()[0][0]])
            for _ in range(frame_repeat):
                self.game.advance_action()
                sleep(0.03)

        self.n_steps += 1
