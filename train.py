from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
from datetime import datetime
import numpy as np
import torch
from tqdm import trange
from models.DQN import DQN
# from models.DRQN import DRQN
import sys

load_model = False
model_name = "ddqn_prioritized_softtarget0.999_defend_the_center"
config_file_path = "scenarios/defend_the_center.cfg"
date = datetime.now().strftime("_%d_%m_%Hh%M")
file_name = "./pretrained_models/"+model_name+".pth"
log_name = "./logs/"+model_name+date
epochs = 30
learning_steps_per_epoch = 5000
episodes_to_watch = 25
best_mean = float("-inf")

# cuda stuff
use_gpu = False
use_ddqn = not False
use_drqn = not True
use_prioritized = True
use_parameter_exploration = False
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
    model = DQN(game, actions, file_name,prioritized=use_prioritized,
                ddqn=use_ddqn, gpu=use_gpu, loading=1)
else:
    model = DQN(game, actions, file_name,prioritized=use_prioritized,
                ddqn=use_ddqn, gpu=use_gpu, parameter_exploration=use_parameter_exploration,
                loading=0)

print("Starting the training.")
time_start = time()
for epoch in range(epochs):

    print("\nEpoch %d\n-------" % (epoch + 1))
    train_episodes_finished = 0

    # training
    print("Training...")
    train_scores = []
    losses = []
    game.new_episode()

    for learning_step in trange(learning_steps_per_epoch, leave=False):

        loss = model.step(training=True)
        losses.append(loss)
        if game.is_episode_finished():
            score = game.get_total_reward()
            train_scores.append(score)
            game.new_episode()
            train_episodes_finished += 1

    train_scores = np.array(train_scores)
    losses = np.array(losses)
    print("%d training episodes played." % train_episodes_finished)
    print("Current size of the memory buffer:", sys.getsizeof(model.memory))
    print("Results: mean score: %.1f +/- %.1f,"
          % (train_scores.mean(), train_scores.std()),
          "min: %.1f," % train_scores.min(), "max: %.1f,"
          % train_scores.max())
    print("Results: mean loss: %.1f +/- %.1f,"
          % (losses.mean(), losses.std()),
          "min: %.1f," % losses.min(), "max: %.1f,"
          % losses.max())

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
    print("Results: mean score: %.1f +/- %.1f,"
          % (test_scores.mean(), test_scores.std()),
          "min: %.1f," % test_scores.min(), "max: %.1f,"
          % test_scores.max())

    if (test_scores.mean() > best_mean):
        best_mean = test_scores.mean()
        model.save()

    # saving results
    with open(log_name+"_loss.txt", "a+") as file:
        file.write(str(losses.mean())+'\n')
    with open(log_name+"_train_score.txt", "a+") as file:
        file.write(str(train_scores.mean())+'\n')
    with open(log_name+"_test_score.txt", "a+") as file:
        file.write(str(test_scores.mean())+'\n')



game.close()
print("======================================")
print("Training finished.")
