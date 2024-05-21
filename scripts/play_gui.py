import gymnasium as gym
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()


model = MaskablePPO.load("models/5x5_MLP-default_lr-0.0003_31")

env = selfplay_wrapper(HexEnv)(play_gui=True, board_size=5, scores=np.zeros(20),prob_model=model)
env = ActionMasker(env, mask_fn)


# env.unwrapped.opponent_models[0].model = model
# env.unwrapped.opponent_models[0].model = model
# env.unwrapped.prob_model = model
state, info = env.reset()
terminated = False


while not terminated:
    board = state

    action, _states = model.predict(state, deterministic=True, action_masks=env.legal_actions())
    print("agent played: ", action)

    state, reward, terminated, truncated, info = env.step(action)