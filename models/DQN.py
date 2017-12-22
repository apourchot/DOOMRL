from vizdoom import *
import itertools as it
from random import sample, randint, random, randint
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
FRAME_REPEAT = 4
EPISODE_LENGTH = 10
LSTM_MEMORY = 512
RESOLUTION = (52, 104)
NB_CHANNELS = 3

# dqn parameters - ok to change won't affect test
NB_EPOCH = 250
DATA_AUGMENTATION = False
REPLAY_MEMORY_SIZE = 7500
BATCH_SIZE = 32
UPDATE_FREQUENCY = 4
Q_NET_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE =  REPLAY_MEMORY_SIZE / 4

# epsilon schedule parameters
EPS_START = 1
EPS_END = 0.1
EPS_LIN_DECAY_START = REPLAY_START_SIZE
EPS_LIN_DECAY_END = REPLAY_START_SIZE * NB_EPOCH

# lr schedule parameters
LR_START = 0.00005

# pytorch tensors
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

# Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
    def sample(self):

        samples = []
        for i in range(BATCH_SIZE):

            # picking random element from memory
            index = randint(0, len(self) - 1)
            state = self.memory[index].state
            next_state = self.memory[index].next_state
            action = self.memory[index].action
            reward = self.memory[index].reward

            # data augmention
            if(DATA_AUGMENTATION):
                u = random()
                if(u >= 1/2):
                    state = flip(state, -1)
                    if(next_state is not None):
                        next_state = flip(next_state, -1)


            # adding to samples
            samples += [Transition(state, action, next_state, reward)]
        batch = Transition(*zip(*samples))

        return batch

    # sample batch_size short episodes of length episode_length from memory
    def sample_episode(self):
        i = 0
        episodes = []
        while(i < BATCH_SIZE):
            frames = torch.Tensor(EPISODE_LENGTH, 1, NB_CHANNELS,
            RESOLUTION[0], RESOLUTION[1]).type(FloatTensor)
            frames_next = torch.Tensor(EPISODE_LENGTH, 1, NB_CHANNELS,
            RESOLUTION[0], RESOLUTION[1]).type(FloatTensor)
            begin = randint(0, len(self) - EPISODE_LENGTH - 1)

            for j in range(EPISODE_LENGTH):
                # we're about to add the episode in the batch
                if(j == EPISODE_LENGTH - 1):
                    frames[j] = self.memory[begin + j].state
                    is_terminal = (self.memory[begin + j].next_state is None)
                    if(is_terminal):
                        frames_next = None
                    else:
                        frames_next[j] = self.memory[begin + j + 1].state
                    action = self.memory[begin + j].action
                    reward = self.memory[begin + j].reward
                    if(DATA_AUGMENTATION):
                        u = random()
                        if(u >= 1/2):
                            frames = flip(frames, -1)
                            if(frames_next is not None):
                                frames_next = flip(frames_next, -1)
                    episodes += [Transition(frames, action, frames_next, reward)]
                    i += 1

                # we jumped to another scenario, we reject the sample
                elif(self.memory[begin + j].next_state is None):
                    break

                # onto next fram
                else:
                    frames[j] = self.memory[begin + j].state
                    frames_next[j] = self.memory[begin + j + 1].state

        batch = Transition(*zip(*episodes))
        return batch

    def __len__(self):
        return len(self.memory)


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
                gpu=False, loading=False):

        # misc
        self.game = game_instance
        self.actions = actions
        self.file_name = file_name
        self.n_steps = 1
        self.history_frames = np.zeros((EPISODE_LENGTH, 1, NB_CHANNELS,
                                        RESOLUTION[0], RESOLUTION[1]))

        # DQN memory and DQNetworks
        self.ddqn = ddqn
        self.drqn = drqn
        self.memory = ReplayMemory(capacity=REPLAY_MEMORY_SIZE)
        if(loading):
            print("Loading model from: ", file_name)
            self.q_net = torch.load(file_name)
        else:
            if(self.drqn):
                self.q_net = DRQNet(len(actions))
            else:
                self.q_net = DQNet(len(actions))
        self.q_net_target = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(),
                                             lr=LR_START)

        if(gpu):
            global FloatTensor
            global LongTensor
            global ByteTensor
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
            ByteTensor = torch.cuda.ByteTensor
            self.q_net.cuda()
            self.q_net_target.cuda()

    def append_history(self, state):

        self.history_frames = np.roll(self.history_frames, -1, axis=0)
        self.history_frames[-1] = state[0,:,:,:]

        if(SHOW_IMAGES):
            for i in range(EPISODE_LENGTH):
                save_image(self.history_frames[i,0,:,:,:],
                                "hist_"+str(i)+".png")

    def learn(self, q_values, expected_q_values):

        # computing loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        # gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

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
            return LongTensor([[randint(0, len(self.actions) - 1)]])

    def save(self):

        torch.save(self.q_net, self.file_name)
        print("model saved at:", self.file_name)

    def step(self, training, showing=False):

        # observing state and selecting action with eps-greedy policy
        frame = self.game.get_state().screen_buffer
        preprocessed_frame = preprocess(frame)
        if(self.drqn):
            self.append_history(preprocessed_frame)
        s1 = torch.from_numpy(preprocessed_frame).type(FloatTensor)
        a = self.select_action(s1, training)

        # if we're training the DQNetwork
        if(training):

            # experiencing transition and adding it in memory
            reward = FloatTensor([self.game.make_action(self.actions[a[0][0]], FRAME_REPEAT)])
            is_terminal = self.game.is_episode_finished()
            if(not is_terminal):
                frame = self.game.get_state().screen_buffer
                preprocessed_frame = preprocess(frame)
                if(self.drqn):
                    s2 = torch.from_numpy(preprocessed_frame).type(FloatTensor)
                else:
                    s2 = torch.from_numpy(preprocessed_frame).type(FloatTensor)
            else:
                s2 = None
            self.memory.push(s1, a, s2, reward)

             # learning
            if (len(self.memory) > REPLAY_START_SIZE and
                self.n_steps % UPDATE_FREQUENCY == 0):

                # getting batch
                if(self.drqn):
                    batch = self.memory.sample_episode()
                else:
                    batch = self.memory.sample()

                state_batch = Variable(torch.cat(batch.state, dim=self.drqn))
                action_batch = Variable(torch.cat(batch.action))
                reward_batch = Variable(torch.cat(batch.reward))

                # taking care of final states
                non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None], dim=self.drqn), volatile=True)
                non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

                # computing expected q values with background DQNetwork
                if(self.drqn or self.ddqn):
                    next_v_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
                    arg_max = self.q_net(non_final_next_states).max(1)[1].view(-1, 1)
                    next_v_values[non_final_mask] = self.q_net_target(non_final_next_states).gather(1, arg_max)
                else:
                    next_v_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
                    next_v_values[non_final_mask] = self.q_net_target(non_final_next_states).max(1)[0]
                expected_q_values = next_v_values * DISCOUNT_FACTOR + reward_batch
                expected_q_values.volatile = False

                # computing q values with DQNetwork
                q_values = self.q_net(state_batch).gather(1, action_batch)

                # descent step
                self.learn(q_values, expected_q_values)

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
