import gym
import minihex

from minihex.interactive.interactive import InteractiveGame
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

env = gym.make("hex-v0",
               opponent_policy="NN",
            #    opponent_policy="interactice",
            #    opponent_policy=minihex.random_policy,
               board_size=11, show_board=True)

state, info = env.reset()
done = False
# interactive = InteractiveGame(config, env)

while not done:
    board, player = state
    action = env.interactive_play(board, player, info)
    # action = minihex.random_policy(board, player, info)
    state, reward, done, info = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")
