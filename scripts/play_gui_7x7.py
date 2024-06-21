import gymnasium as gym
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper, BaseRandomPolicy
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()


model = MaskablePPO.load("models/7x7_MLP-default_lr-0.0003_56")

env = selfplay_wrapper(HexEnv)(play_gui=True, 
                               board_size=7, 
                               scores=np.zeros(20),
                               prob_model=model,
                               agent_player_num=0)

env = ActionMasker(env, mask_fn)


state, info = env.reset()
terminated = False


while not terminated:
    board = state

    action, _states = model.predict(state, deterministic=True, action_masks=env.legal_actions())
    print("agent played: ", action)

    state, reward, terminated, truncated, info = env.step(action)