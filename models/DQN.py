from vizdoom import *
import itertools as it
from random import sample, randint, random, randint, uniform
from collections import namedtuple
from time import time, sleep
from PIL import Image
from time import time
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from .utils.ReplayMemory import ReplayMemory
import torchvision.transforms as T
import skimage.transform
import skimage.color
import skimage.io
import skimage
from scipy.misc import imresize, imsave
from tqdm import trange

# to save images
SHOW_IMAGES = False

# dqn parameters - not ok to change will affect test
DISCOUNT_FACTOR = 0.99
FRAME_REPEAT = 8
EPISODE_LENGTH = 4
LSTM_MEMORY = 512
RESOLUTION = (52, 104)
NB_CHANNELS = 3

# dqn parameters - ok to change won't affect test
NB_EPOCH = 50
DATA_AUGMENTATION = False
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
UPDATE_FREQUENCY = 4
Q_NET_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 2500

# experience replay parameters
ALPHA = 0.6
BETA_START = 0.4
BETA_END = 1
BETA_LIN_DECAY_START = REPLAY_START_SIZE
BETA_LIN_DECAY_END = REPLAY_START_SIZE * NB_EPOCH

# epsilon schedule parameters
EPS_START = 1
EPS_END = 0.1
EPS_LIN_DECAY_START = REPLAY_START_SIZE
EPS_LIN_DECAY_END = REPLAY_START_SIZE * NB_EPOCH

# lr schedule parameters
LR_START = 0.00025

# pytorch tensors
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

# Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
TransitionV2 = namedtuple('TransitionV2', ('state', 'action', 'next_state', 'reward', 'index', 'weight'))

# https://github.com/pytorch/pytorch/issues/229
def flip(x, dim):

    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))

    return x[inds]

def save_image(img, file_name):

    img = img.squeeze()
    img = np.moveaxis(img, 0, -1)
    imsave(file_name, img)

def preprocess(img):

    img = imresize(img, RESOLUTION)
    img = np.reshape(img, (1, RESOLUTION[0], RESOLUTION[1], NB_CHANNELS))
    img = np.moveaxis(img, -1, 1)

    return img


class DQNet(nn.Module):

    def __init__(self, nb_available_actions):

        super(DQNet, self).__init__()

        # conv layers
        self.conv1 = nn.Conv2d(NB_CHANNELS, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3,stride=1)

        # fc layers
        self.fc1 = nn.Linear(1728, 512)
        self.fc2 = nn.Linear(512, nb_available_actions)

        # batch norm
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)

        xavier_init = torch.nn.init.xavier_uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                xavier_init(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):

        LReLU = F.leaky_relu

        x = x / 255.

        # conv layers
        x = LReLU(self.conv1(x))
        x = self.batch_norm1(x)
        x = LReLU(self.conv2(x))
        x = self.batch_norm2(x)
        x = LReLU(self.conv3(x))

        # fc layers
        x = x.view(x.size(0), -1)
        x = LReLU(self.fc1(x))
        return self.fc2(x)


class DRQNet(nn.Module):

    def __init__(self, nb_available_actions):

        super(DRQNet, self).__init__()

        # conv layers
        self.conv1 = nn.Conv2d(NB_CHANNELS, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3,stride=1)

        # lstm
        self.lstm = nn.LSTM(1728, LSTM_MEMORY, 1)

        # fc layers
        self.fc = nn.Linear(LSTM_MEMORY, nb_available_actions)

        # batch norm
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)

        xavier_init = torch.nn.init.xavier_uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                xavier_init(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):

        LReLU = F.leaky_relu
        hidden_state = Variable(torch.zeros(1, x.size(1), LSTM_MEMORY).type(FloatTensor))
        cell_state = Variable(torch.zeros(1, x.size(1), LSTM_MEMORY).type(FloatTensor))

        x = x / 255.

        # conv layers
        z = Variable(torch.zeros(x.size(0), x.size(1), 1728)).type(FloatTensor)
        for k in range(x.size(0)):
            y = LReLU(self.conv1(x[k]))
            y = self.batch_norm1(y)
            y = LReLU(self.conv2(y))
            y = self.batch_norm2(y)
            y = LReLU(self.conv3(y))
            z[k] = y.view(y.size(0), 1728)

        # lstm layer
        z, (hidden_state, cell_state) = self.lstm(z, (hidden_state, cell_state))

        # linear layer
        z = z[-1]
        return self.fc(z)


class DQN():

    def __init__(self, game_instance, actions, file_name, ddqn=False, drqn=False,
                prioritized=False, gpu=False, loading=False):

        # misc
        self.game = game_instance
        self.actions = actions
        self.file_name = file_name
        self.n_steps = 1
        self.history_frames = np.zeros((EPISODE_LENGTH, 1, NB_CHANNELS,
                                        RESOLUTION[0], RESOLUTION[1]))

        # DQNetworks
        self.ddqn = ddqn
        self.drqn = drqn
        if(loading):
            print("Loading model from: ", file_name)
            self.q_net = torch.load(file_name, map_location=lambda storage, loc: storage)
        else:
            if(self.drqn):
                self.q_net = DRQNet(len(actions))
            else:
                self.q_net = DQNet(len(actions))
        self.q_net_target = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                            betas=(0.9, 0.999),
                                            eps=10**-8,
                                            lr=LR_START)

        # if we're using gpu
        if(gpu):
            global FloatTensor
            global LongTensor
            global ByteTensor
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
            ByteTensor = torch.cuda.ByteTensor
            self.q_net.cuda()
            self.q_net_target.cuda()

        # DQN memory
        self.prioritized = prioritized
        self.memory = ReplayMemory(capacity=REPLAY_MEMORY_SIZE,
                                    prioritized=prioritized)


    def append_history(self, state):

        self.history_frames = np.roll(self.history_frames, -1, axis=0)
        self.history_frames[-1] = state[0,:,:,:]

        if(SHOW_IMAGES):
            for i in range(EPISODE_LENGTH):
                save_image(self.history_frames[i,0,:,:,:],
                                "hist_"+str(i)+".png")

    def learn(self, q_values, expected_q_values, weights_IS=1):

        # computing loss
        # batch_loss = (torch.abs(q_values - expected_q_values) < 1).float() *\
        #              (q_values - expected_q_values) ** 2 +\
        #              (torch.abs(q_values - expected_q_values) >= 1).float() *\
        #              (torch.abs(q_values - expected_q_values) - 0.5)
        batch_loss = (q_values - expected_q_values) ** 2
        weighted_batch_loss = weights_IS * batch_loss
        weighted_loss = weighted_batch_loss.sum()

        # gradient descent step
        self.optimizer.zero_grad()
        weighted_loss.backward()
        for p in self.q_net.parameters():
            p.grad.data.clamp_(-10, 10)
        self.optimizer.step()

        return weighted_loss

    def select_action(self, state, training):

        # epsilon greedy policy with epsilon exponentially decaying during
        # training process
        u = random()
        eps = max(EPS_END, EPS_START + (self.n_steps - EPS_LIN_DECAY_START) *
                                        (EPS_END - EPS_START) /
                                        (EPS_LIN_DECAY_END - EPS_LIN_DECAY_START))
        if (u > eps or not training):
            if(self.drqn):
                torch_history = torch.from_numpy(self.history_frames)
                return self.q_net(Variable(torch_history, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
            else:
                return self.q_net(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[randint(0, len(self.actions) - 1)]])

    def save(self):

        torch.save(self.q_net, self.file_name)
        print("model saved at:", self.file_name)

    def step(self, training, showing=False):

        # observing state and selecting action with eps-greedy policy
        frame = self.game.get_state().screen_buffer
        preprocessed_frame = preprocess(frame)
        if(self.drqn):
            self.append_history(preprocessed_frame)
        s1 = torch.from_numpy(preprocessed_frame).type(torch.FloatTensor)
        a = self.select_action(s1, training)

        # if we're training the DQNetwork
        if(training):

            # experiencing transition and adding it in memory
            reward = torch.FloatTensor([self.game.make_action(self.actions[a[0][0]], FRAME_REPEAT)])
            is_terminal = self.game.is_episode_finished()
            if(not is_terminal):
                frame = self.game.get_state().screen_buffer
                preprocessed_frame = preprocess(frame)
                s2 = torch.from_numpy(preprocessed_frame).type(torch.FloatTensor)
            else:
                s2 = None
            self.memory.push(s1, a, s2, reward)

             # learning
            if (self.n_steps > REPLAY_START_SIZE and
                self.n_steps % UPDATE_FREQUENCY == 0):

                # getting batch
                if(self.drqn):
                    batch = self.memory.sample_episode()
                else:
                    batch = self.memory.sample()

                state_batch = Variable(torch.cat(batch.state, dim=self.drqn).type(FloatTensor))
                action_batch = Variable(torch.cat(batch.action).type(LongTensor))
                reward_batch = Variable(torch.cat(batch.reward).type(FloatTensor))

                # for i in range(state_batch.size()[0]):
                #     save_image(state_batch[i,10,:,:,:].data.cpu().numpy(), "batch_"+str(i)+".png")

                # taking care of final states
                non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None],
                                                dim=self.drqn).type(FloatTensor))
                non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

                # computing expected q values with background DQNetwork
                if(self.drqn or self.ddqn):
                    next_v_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
                    arg_max = self.q_net(non_final_next_states).detach().max(1)[1].view(-1, 1)
                    next_v_values[non_final_mask] = self.q_net_target(non_final_next_states).detach().gather(1, arg_max)
                else:
                    next_v_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
                    next_v_values[non_final_mask] = self.q_net_target(non_final_next_states).detach().max(1)[0]
                expected_q_values = next_v_values * DISCOUNT_FACTOR + reward_batch
                expected_q_values.volatile = False

                # computing q values with DQNetwork
                q_values = self.q_net(state_batch).gather(1, action_batch).squeeze()

                # computing errors and weigths for experience replay
                weights_IS = 1
                if(self.prioritized):
                    BETA = min(BETA_END, BETA_START + (self.n_steps - BETA_LIN_DECAY_START) *
                                                    (BETA_END - BETA_START) /
                                                    (BETA_LIN_DECAY_END - BETA_LIN_DECAY_START))
                    errors = torch.abs(q_values - expected_q_values) ** ALPHA
                    weights_IS = Variable(torch.cat(batch.weight).type(FloatTensor)) ** BETA
                    weights_IS = torch.sqrt(weights_IS / weights_IS.max())
                    indexes = torch.cat(batch.index)
                    self.memory.update_prior(indexes, errors)

                # descent step
                self.learn(q_values, expected_q_values, weights_IS)

                # updating the background DQNetwork every
                # Q_NET_UPDATE_FREQUENCY step
                if((self.n_steps % Q_NET_UPDATE_FREQUENCY == 0) and
                    self.n_steps >= 2 * REPLAY_START_SIZE):
                    self.q_net_target = copy.deepcopy(self.q_net)

            self.n_steps += 1

        # otherwise we just simulate the action
        else:
            if(showing):
                self.game.set_action(self.actions[a[0][0]])
                for _ in range(FRAME_REPEAT):
                    self.game.advance_action()
            else:
                self.game.make_action(self.actions[a[0][0]], FRAME_REPEAT)
