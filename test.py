from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
from tqdm import trange
from models.DQN import DQN

config_file_path = "scenarios/basic.cfg"
file_name = "./pretrained_models/dqn_basic.pth"
episodes_to_watch = 10


# Creates and initializes ViZDoom environment.
def vizdoom_init(config_file_path):

    print("Initializing DOOM...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


# Create Doom instance
game = vizdoom_init(config_file_path)
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
print("Loading model from: ", file_name)
model = DQN(game, actions, file_name, loading=1)

print("======================================")
print("Testing trained neural network.")

print("Testing...")
test_scores = []

for _ in range(episodes_to_watch):

    game.new_episode()
    while not game.is_episode_finished():
        model.step(training=False)

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    test_scores.append(score)
    print("Total score: ", score)

test_scores = np.array(test_scores)
print("%d test episodes played." % episodes_to_watch)
print("Results: mean: %.1f +/- %.1f,"
      % (train_scores.mean(), train_scores.std()),
      "min: %.1f," % train_scores.min(), "max: %.1f,"
      % train_scores.max())
