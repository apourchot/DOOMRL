from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import torch
from tqdm import trange
from models.DQN import DQN
import sys

load_model = False
config_file_path = "scenarios/take_cover.cfg"
file_name = "./pretrained_models/drqn_prioritized_take_cover.pth"
epochs = 100
learning_steps_per_epoch = 10000
episodes_to_watch = 100
best_mean = float("-inf")

# cuda stuff
use_gpu = False
use_ddqn = False
use_drqn = True
use_prioritized = True
if(torch.cuda.is_available()):
    print("Using GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
    use_gpu = True


# Creates and initializes ViZDoom environment.
def vizdoom_init(config_file_path):

    print("Initializing DOOM...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_256X144)
    game.init()
    print("Doom initialized.")
    return game


# Create Doom instance
game = vizdoom_init(config_file_path)
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

if load_model:
    model = DQN(game, actions, file_name, drqn=use_drqn,
    prioritized=use_prioritized, ddqn=use_ddqn, gpu=use_gpu, loading=1)
else:
    model = DQN(game, actions, file_name, drqn=use_drqn,
    prioritized=use_prioritized, ddqn=use_ddqn, gpu=use_gpu, loading=0)

print("Starting the training.")
time_start = time()
for epoch in range(epochs):

    print("\nEpoch %d\n-------" % (epoch + 1))
    train_episodes_finished = 0

    # training
    print("Training...")
    train_scores = []
    game.new_episode()

    for learning_step in trange(learning_steps_per_epoch, leave=False):

        model.step(training=True)
        if game.is_episode_finished():
            score = game.get_total_reward()
            train_scores.append(score)
            game.new_episode()
            train_episodes_finished += 1

    train_scores = np.array(train_scores)
    print("%d training episodes played." % train_episodes_finished)
    print("Current size of the memory buffer:", sys.getsizeof(model.memory))
    print("Results: mean: %.1f +/- %.1f,"
          % (train_scores.mean(), train_scores.std()),
          "min: %.1f," % train_scores.min(), "max: %.1f,"
          % train_scores.max())

    # testing to assess performances
    print("Testing...")
    test_scores = []

    for _ in trange(episodes_to_watch, leave=False):

        game.new_episode()
        while not game.is_episode_finished():
            model.step(training=False)

        # Sleep between episodes
        score = game.get_total_reward()
        test_scores.append(score)

    test_scores = np.array(test_scores)
    print("%d test episodes played." % episodes_to_watch)
    print("Results: mean: %.1f +/- %.1f,"
          % (test_scores.mean(), test_scores.std()),
          "min: %.1f," % test_scores.min(), "max: %.1f,"
          % test_scores.max())

    if (test_scores.mean() > best_mean and
        epoch >= 1):
        best_mean = test_scores.mean()
        model.save()

game.close()
print("======================================")
print("Training finished.")
