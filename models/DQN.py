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
FRAME_REPEAT = 4
RESOLUTION = (60, 108)
NB_CHANNELS = 3

# dqn parameters - ok to change won't affect test
NB_EPOCH = 30#
DATA_AUGMENTATION = False
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
UPDATE_FREQUENCY = 4
DROPOUT_RATIO = 0.5
USE_BATCH_NORM = True
Q_NET_UPDATE_FREQUENCY = 20000
LAG = 0.999
REPLAY_START_SIZE = 5000

# experience replay parameters
ALPHA = 0.7
BETA_START = 0.5
BETA_END = 1
BETA_LIN_DECAY_START = REPLAY_START_SIZE
BETA_LIN_DECAY_END = REPLAY_START_SIZE * NB_EPOCH

# epsilon schedule parameters
EPS_START = 0.9
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

def get_game_variables(game, variables):

    result = []
    for variable in variables:
        if variable not in [GameVariable.POSITION_X, GameVariable.POSITION_Y]:
            result.append(game.get_game_variable(variable))

    return [result]


def save_image(img, file_name):

    img = img.squeeze()
    img = np.moveaxis(img, 0, -1)
    imsave(file_name, img)

def preprocess(img, is_drqn=False):

    img = imresize(img, RESOLUTION)
    img = np.reshape(img, (1, RESOLUTION[0], RESOLUTION[1], NB_CHANNELS))
    img = np.moveaxis(img, -1, 1)
    img = img.astype(np.float32)
    img = img / 255.0

    return img

def DQN_distance(q_values_1, q_values_2):
    """Compute KL divergence between the two q network values"""
    pi_1 = softmax(q_values_1)
    pi_2 = softmax(q_values_2)
    KL = np.sum(pi_1 * log(pi_1/pi_2))
    return KL

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def repeat(tensor, n_repeat):

    size = tensor.size()
    if(len(size) == 1):
        return tensor.view(-1, 1).repeat(1, n_repeat).view(-1, 1).squeeze()
    else:
        return tensor.repeat(1, n_repeat).view(-1, 1)


class DQNet(nn.Module):

    def __init__(self, nb_available_actions, nb_game_variables):

        super(DQNet, self).__init__()

        # conv layers
        self.conv1 = nn.Conv2d(NB_CHANNELS, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3,stride=1)

        # fc layers
        self.fc1 = nn.Linear(2560+nb_game_variables, 512)
        self.fc2 = nn.Linear(512, nb_available_actions)

        # batch norm
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)

        # dropout
        self.dropout = nn.Dropout(p=DROPOUT_RATIO)

        xavier_init = torch.nn.init.xavier_uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                xavier_init(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x, game_variables=None, train=True):

        LReLU = F.leaky_relu

        if(not train):
            self.dropout.eval()

        else:
            self.dropout.train()

        # conv layers
        x = LReLU(self.conv1(x))
        if USE_BATCH_NORM:
            x = self.batch_norm1(x)
        x = LReLU(self.conv2(x))
        if USE_BATCH_NORM:
            x = self.batch_norm2(x)
        x = LReLU(self.conv3(x))

        # fc Batch
        x = x.view(x.size(0), -1)
        x = torch.cat((x, game_variables), dim=1)
        x = LReLU(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class DQN():

    def __init__(self, game_instance, actions, file_name, ddqn=False,
                prioritized=False, parameter_exploration=False, gpu=False, loading=False):

        # misc
        self.game = game_instance
        self.actions = actions
        self.file_name = file_name
        self.available_game_variables = game_instance.get_available_game_variables()
        self.nb_game_variables = len([v for v in self.available_game_variables
                                     if v not in [GameVariable.POSITION_X,
                                                  GameVariable.POSITION_Y]])
        self.n_steps = 1

        # DQNetworks
        self.ddqn = ddqn
        self.parameter_exploration = parameter_exploration
        if(loading):
            print("Loading model from: ", file_name)
            self.q_net = torch.load(file_name, map_location=lambda storage, loc: storage)
        else:
            self.q_net = DQNet(len(actions),
                                    self.nb_game_variables)
        self.q_net_target = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                            betas=(0.9, 0.999),
                                            eps=10**-8,
                                            lr=LR_START)
        # self.optimizer = torch.optim.RMSprop(self.q_net.parameters(),
        #                                      lr=LR_START)

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
                                   prioritized=prioritized,
                                   nb_game_variables=self.nb_game_variables)


    def append_history(self, state):

        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = state[0,:,:,:]

        # if(SHOW_IMAGES):
        #     for i in range(EPISODE_LENGTH):
        #         save_image(self.history[i,0,:,:,:],
        #                         "hist_"+str(i)+".png")

    def reset_history(self):

        if(self.drqn):
            self.history = np.zeros((EPISODE_LENGTH, 1, NB_CHANNELS, RESOLUTION[0],
                                RESOLUTION[1]))
        else:
            self.history = np.zeros((1, 1, NB_CHANNELS, RESOLUTION[0],
                                RESOLUTION[1]))

    def learn(self, q_values, expected_q_values, weights_IS=1):

        # computing loss
        batch_loss = (q_values - expected_q_values) ** 2
        weighted_batch_loss = weights_IS * batch_loss
        weighted_loss = 1/BATCH_SIZE * weighted_batch_loss.sum()

        # gradient descent step
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # for p in self.q_net.parameters():
        #     p.grad.data.clamp_(-5, 5)
        self.optimizer.step()

        return weighted_loss.data

    def select_action(self, s, v, training=False):

        # epsilon greedy policy with epsilon exponentially decaying during
        # training process
        u = random()
        eps = max(EPS_END, EPS_START + (self.n_steps - EPS_LIN_DECAY_START) *
                                        (EPS_END - EPS_START) /
                                    (EPS_LIN_DECAY_END - EPS_LIN_DECAY_START))

        if(not self.parameter_exploration):
            if (u > eps or not training):
                s_ = Variable(s, volatile=True).type(FloatTensor)
                v_ = Variable(v, volatile=True).type(FloatTensor)
                return self.q_net(s_, v_, train=False).data.max(1)[1].view(1, 1)
            else:
                return torch.LongTensor([[randint(0, len(self.actions) - 1)]])
        else:
            delta = -np.log(1 - eps + eps/self.nb_available_actions)
            if(d < delta):
                self.sigma *= 1.01
            else:
                self.sigma /= 1.01

    def save(self):

        torch.save(self.q_net, self.file_name)
        print("model saved at:", self.file_name)

    def step(self, training, showing=False):

        # observing state and selecting action with eps-greedy policy
        frame = self.game.get_state().screen_buffer
        preprocessed_frame = preprocess(frame)
        # save_image(preprocessed_frame, "frame.jpg")
        s1 = torch.from_numpy(preprocessed_frame).type(torch.FloatTensor)
        v1 = torch.FloatTensor(get_game_variables(self.game, self.available_game_variables))
        a = self.select_action(s=s1, v=v1, training=training)
        loss = 0

        # if we're training the DQNetwork
        if(training):

            # experiencing transition and adding it in memory
            reward = torch.FloatTensor([self.game.make_action(self.actions[a[0][0]], FRAME_REPEAT)])
            is_terminal = self.game.is_episode_finished()
            if(not is_terminal):
                frame = self.game.get_state().screen_buffer
                preprocessed_frame = preprocess(frame)
                s2 = torch.from_numpy(preprocessed_frame).type(torch.FloatTensor)
                v2 = torch.FloatTensor(get_game_variables(self.game, self.available_game_variables))
            else:
                s2 = None
                v2 = None


            self.memory.push(s1, v1, a, s2, v2, reward)

             # learning
            if (self.n_steps > REPLAY_START_SIZE and
                self.n_steps % UPDATE_FREQUENCY == 0):

                # getting sample from memory
                batch = self.memory.sample(BATCH_SIZE)

                # batch content
                state_batch = Variable(torch.cat(batch.state).type(FloatTensor))
                variable_batch = Variable(torch.cat(batch.variable).type(FloatTensor))
                action_batch = Variable(torch.cat(batch.action).type(LongTensor))
                reward_batch = Variable(torch.cat(batch.reward).type(FloatTensor))
                is_weights_batch = Variable((torch.cat(batch.weight) if self.prioritized else torch.zeros(BATCH_SIZE)).type(FloatTensor))
                index_batch = torch.cat(batch.index) if self.prioritized else torch.zeros(BATCH_SIZE)

                # taking care of final states
                non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]).type(FloatTensor))
                non_final_next_variables = Variable(torch.cat([v for v in batch.next_variable if v is not None]).type(FloatTensor))
                non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

                # computing expected q values with background DQNetwork
                if(self.ddqn):
                    next_v_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor), volatile=True)
                    arg_max = self.q_net(non_final_next_states, non_final_next_variables).detach().max(1)[1].view(-1, 1)
                    next_v_values[non_final_mask] = self.q_net_target(non_final_next_states, non_final_next_variables).detach().gather(1, arg_max)
                else:
                    next_v_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor), volatile=True)
                    next_v_values[non_final_mask] = self.q_net_target(non_final_next_states, non_final_next_variables).detach().max(1)[0]
                expected_q_values = next_v_values * DISCOUNT_FACTOR + reward_batch
                expected_q_values.volatile = False

                # computing q values with DQNetwork
                q_values = self.q_net(state_batch, variable_batch).gather(1, action_batch).squeeze()

                # computing errors and weigths for prioritized experience replay
                if(self.prioritized):
                    errors = torch.abs(q_values - expected_q_values) ** ALPHA
                    BETA = min(BETA_END, BETA_START + (self.n_steps - BETA_LIN_DECAY_START) *
                                                    (BETA_END - BETA_START) /
                                                    (BETA_LIN_DECAY_END - BETA_LIN_DECAY_START))
                    is_weights_batch = is_weights_batch ** BETA
                    is_weights_batch = is_weights_batch / is_weights_batch.max()
                    self.memory.update_prior(index_batch, errors)

                # descent step
                loss = self.learn(q_values, expected_q_values, is_weights_batch)

                # updating the background DQNetwork every
                # Q_NET_UPDATE_FREQUENCY step
                # if((self.n_steps % Q_NET_UPDATE_FREQUENCY == 0) and
                #     self.n_steps >= 2 * REPLAY_START_SIZE):
                #     self.q_net_target = copy.deepcopy(self.q_net)

                for p, q in zip(self.q_net_target.parameters(), self.q_net.parameters()):
                    p.data = LAG * copy.deepcopy(p.data) + (1 - LAG) * copy.deepcopy(q.data)

            self.n_steps += 1

        # otherwise we just simulate the action
        else:
            if(showing):
                self.game.set_action(self.actions[a[0][0]])
                for _ in range(FRAME_REPEAT):
                    self.game.advance_action()
            else:
                self.game.make_action(self.actions[a[0][0]], FRAME_REPEAT)


        return loss
