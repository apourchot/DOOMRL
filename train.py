from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
from tqdm import trange
from models.DQN import DQN

load_model = False
config_file_path = "scenarios/rocket_basic.cfg"
file_name = "./pretrained_models/dqn_rocket_basic.pth"
epochs = 50
learning_steps_per_epoch = 2000


# Creates and initializes ViZDoom environment.
def vizdoom_init(config_file_path):

    print("Initializing DOOM...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


# Create Doom instance
game = vizdoom_init(config_file_path)
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

if load_model:
    print("Loading model from: ", file_name)
    model = DQN(game, actions, file_name, loading=1, training=1)
else:
    model = DQN(game, actions, file_name, loading=0, training=1)


print("Starting the training.")
time_start = time()
for epoch in range(epochs):
    print("\nEpoch %d\n-------" % (epoch + 1))
    train_episodes_finished = 0
    train_scores = []

    print("Training...")
    game.new_episode()
    for learning_step in trange(learning_steps_per_epoch, leave=False):
        model.step()
        if game.is_episode_finished():
            score = game.get_total_reward()
            train_scores.append(score)
            game.new_episode()
            train_episodes_finished += 1

    print("%d training episodes played." % train_episodes_finished)

    train_scores = np.array(train_scores)

    print("Results: mean: %.1f +/- %.1f,"
          % (train_scores.mean(), train_scores.std()),
          "min: %.1f," % train_scores.min(), "max: %.1f,"
          % train_scores.max())

game.close()
print("======================================")
print("Training finished.")
