import gym
import minihex

from minihex.interactive.interactive import InteractiveGame
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

env = gym.make("hex-v0",
               opponent_policy="interactive",
               board_size=11, show_board=True)

state, info = env.reset()
done = False
# interactive = InteractiveGame(config, env)

while not done:
    print("while")
    # breakpoint()
    board, player = state
    # breakpoint()
    # interactive.board = board
    # interactive.gui.update_board(board)
    # action = interactive.play_move()
    # print(action)
    # action = env.simulator.coordinate_to_action(action)
    # winner = env.simulator.fast_move(action)
    action = minihex.random_policy(board, player, info)
    state, reward, done, info = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")
