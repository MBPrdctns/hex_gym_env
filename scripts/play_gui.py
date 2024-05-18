import gymnasium as gym
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()

env = selfplay_wrapper(HexEnv)(play_gui=True, board_size=5, scores=np.zeros(20))
env = ActionMasker(env, mask_fn)

model = MaskablePPO.load("hex_selfplay_new_10M_5x5")

state, info = env.reset()
terminated = False

while not terminated:
    board = state

    action, _states = model.predict(state, deterministic=True, action_masks=env.legal_actions())
    print("agent played: ", action)

    env.unwrapped.opponent_model.model = model
    state, reward, terminated, truncated, info = env.step(action)