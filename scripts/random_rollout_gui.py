import gymnasium as gym
import minihex
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from minihex.interactive.interactive import InteractiveGame
from configparser import ConfigParser
from minihex.HexGame import player as hex_player


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()


config = ConfigParser()
config.read('config.ini')

env = gym.make("hex-v0",
                opponent_policy="interactive",
                player_color=hex_player.BLACK,
                # current_player_num = hex_player.WHITE,
                board_size=5, show_board=True)
env = ActionMasker(env, mask_fn)
model = MaskablePPO.load("hex_selfplay_history")
state, info = env.reset()
terminated = False
# interactive = InteractiveGame(config, env)

while not terminated:
    print("while")
    # breakpoint()
    board = state
    # breakpoint()
    # interactive.board = board
    # interactive.gui.update_board(board)
    # action = interactive.play_move()
    # print(action)
    # action = env.simulator.coordinate_to_action(action)
    # winner = env.simulator.fast_move(action)
    action, _states = model.predict(state, deterministic=True, action_masks=env.get_action_mask())
    state, reward, terminated, truncated, info = env.step(action) # minihex.random_policy(board, player, info)
    # state, reward, done, info = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")
