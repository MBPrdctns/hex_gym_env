import gymnasium as gym
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper, BaseRandomPolicy
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()


env = selfplay_wrapper(HexEnv)(play_gui=True, board_size=11, scores=np.zeros(20))#,prob_model=model)
env = ActionMasker(env, mask_fn)

state, info = env.reset()
terminated = False


while not terminated:
    board = state

    # action, _states = model.predict(state, deterministic=True, action_masks=env.legal_actions())
    action = BaseRandomPolicy().choose_action(board=board)
    print("agent played: ", action)

    state, reward, terminated, truncated, info = env.step(action)