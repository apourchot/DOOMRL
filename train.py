from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
from tqdm import trange
from models.DQN import DQN

load_model = True
config_file_path = "scenarios/basic.cfg"
file_name = "./pretrained_models/dqn_basic.pth"
epochs = 50
learning_steps_per_epoch = 2000
episodes_to_watch = 5


# Creates and initializes ViZDoom environment.
def vizdoom_init(config_file_path):

    print("Initializing DOOM...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
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
    model = DQN(game, actions, file_name, loading=1)
else:
    model = DQN(game, actions, file_name, loading=0)


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
        sleep(1.0)
        score = game.get_total_reward()
        test_scores.append(score)

    test_scores = np.array(test_scores)
    print("%d test episodes played." % episodes_to_watch)
    print("Results: mean: %.1f +/- %.1f,"
          % (test_scores.mean(), test_scores.std()),
          "min: %.1f," % test_scores.min(), "max: %.1f,"
          % test_scores.max())

game.close()
print("======================================")
print("Training finished.")
