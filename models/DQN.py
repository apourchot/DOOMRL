from vizdoom import *
import itertools as it
from random import sample, randint, random
from collections import namedtuple
from time import time, sleep
from PIL import Image
import numpy as np
import copy
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
from scipy.misc import imresize, imsave
from tqdm import trange


# dqn parameters
discount_factor = 0.99
replay_memory_size = 10000
history_length = 1
resolution = (44, 84)
batch_size = 64
frame_repeat = 8
update_frequency = 1
nb_channels = 3
q_net_update_frequency = 2500
replay_start_size = 2500
nb_epoch = 50

# epsilon schedule parameters
eps_start = 1
eps_end = 0.1
eps_lin_decay_start = replay_start_size
eps_lin_decay_end = replay_start_size * nb_epoch

# lr schedule parameters
lr_start = 0.00025
lr_end = 0.000025
lr_lin_decay_start = replay_start_size
lr_lin_decay_end = replay_start_size * nb_epoch

def preprocess(img):

    img = imresize(img, resolution)
    img = np.reshape(img, (resolution[0], resolution[1], nb_channels))
    img = np.moveaxis(img, -1, 0)

    return img

# preprocess = T.Compose([T.ToPILImage(), T.Resize(resolution, interpolation=Image.CUBIC), T.ToTensor()])
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # add a transition in memory
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # sample a batch from memory
    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):

    def __init__(self, nb_available_actions):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(history_length * nb_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3,stride=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(896, 512)
        self.fc2 = nn.Linear(512, nb_available_actions)

    def forward(self, x):

        LReLU = F.leaky_relu
        x = LReLU(self.conv1(x))
        x = self.batch_norm1(x)
        x = LReLU(self.conv2(x))
        x = self.batch_norm2(x)
        x = LReLU(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = LReLU(self.fc1(x))
        return self.fc2(x)


class DQN():

    def __init__(self, game_instance, actions, file_name, loading):

        # misc
        self.game = game_instance
        self.actions = actions
        self.file_name = file_name
        self.n_steps = 1
        self.history_frames = np.zeros((1, history_length * nb_channels,
                                        resolution[0], resolution[1]))

        # DQN memory and networks
        self.memory = ReplayMemory(capacity=replay_memory_size)
        if(loading):
            print("Loading model from: ", file_name)
            self.q_net = torch.load(file_name)
        else:
            self.q_net = Net(len(actions))
        self.q_net_ = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(),
                                             lr=lr_start)

    def append_history(self, state):

        state = preprocess(state)
        for i in range(nb_channels):
            self.history_frames = np.roll(self.history_frames, -1, axis=1)
            self.history_frames[0,-1,:,:] = state[i]

        for i in range(history_length * nb_channels):
            img = self.history_frames[0,i,:,:]
            imsave("hist_"+str(i)+".png", img)

    def learn(self, q_values, expected_q_values):

        lr = max(lr_end, lr_start + (self.n_steps - lr_lin_decay_start) *
                                        (lr_end - lr_start) /
                                        (lr_lin_decay_end - lr_lin_decay_start))

        # computing loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        # gradient descent step
        self.optimizer.zero_grad()
        self.optimizer.lr = lr
        loss.backward()
        self.optimizer.step()

        return loss

    def select_action(self, state, training):

        # epsilon greedy policy with epsilon exponentially decaying during
        # training process
        u = random()
        eps = max(eps_end, eps_start + (self.n_steps - eps_lin_decay_start) *
                                        (eps_end - eps_start) /
                                        (eps_lin_decay_end - eps_lin_decay_start))
        if (u > eps or not training):
            return self.q_net(Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[randint(0, len(self.actions) - 1)]])

    def save(self):

        torch.save(self.q_net, self.file_name)
        print("model saved at:", self.file_name)

    def step(self, training, showing=False):

        frame = self.game.get_state().screen_buffer
        self.append_history(frame)
        s1 = torch.from_numpy(self.history_frames).type(torch.FloatTensor)
        a = self.select_action(s1, training)

        # if we're training the network
        if(training):

            # experiencing transition and adding it in memory
            reward = torch.FloatTensor([self.game.make_action(
                                self.actions[a.numpy()[0][0]], frame_repeat)])
            is_terminal = self.game.is_episode_finished()
            if(not is_terminal):
                frame = self.game.get_state().screen_buffer
                self.append_history(frame)
                s2 = torch.from_numpy(self.history_frames).type(torch.FloatTensor)
            else:
                s2 = None
            self.memory.push(s1, a, s2, reward)

            # learning
            if (len(self.memory) >= replay_start_size and
                self.n_steps % update_frequency == 0):

                # to get terminal states
                transitions = self.memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                # getting batch from memory
                state_batch = Variable(torch.cat(batch.state))
                action_batch = Variable(torch.cat(batch.action))
                reward_batch = Variable(torch.cat(batch.reward))

                # taking care of final states
                non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                      batch.next_state)))
                non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]),
                                                volatile=True)

                # computing expected q values with background network
                next_v_values = Variable(torch.zeros(batch_size).type(torch.FloatTensor))
                next_v_values[non_final_mask] = self.q_net_(non_final_next_states).max(1)[0]
                next_v_values.volatile = False
                expected_q_values = next_v_values * discount_factor + reward_batch

                # computing q values with network
                q_values = self.q_net(state_batch).gather(1, action_batch)

                # descent step
                self.learn(q_values, expected_q_values)

                # updating the background network every
                # q_net_update_frequency step
                if((self.n_steps % q_net_update_frequency == 0) and
                    self.n_steps >= 2 * replay_start_size):
                    self.q_net_ = copy.deepcopy(self.q_net)

            self.n_steps += 1

        # otherwise we just simulate the action
        else:
            if(showing):
                self.game.set_action(self.actions[a.numpy()[0][0]])
                for _ in range(frame_repeat):
                    self.game.advance_action()
            else:
                self.game.make_action(self.actions[a.numpy()[0][0]], frame_repeat)
