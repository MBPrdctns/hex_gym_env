import gym
import minihex
import numpy as np
from minihex import player


def random_policy(board, player):
    # never surrender :)
    idx = minihex.empty_tiles(board)
    choice = np.random.randint(len(idx))
    return idx[choice]


env = gym.make("hex-v0", opponent_policy=random_policy)

import tqdm
for _ in tqdm.tqdm(range(10000)):
    state = env.reset()
    done = False
    while not done:
        board, player = state
        action = random_policy(board, player)
        state, reward, done, _ = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")
