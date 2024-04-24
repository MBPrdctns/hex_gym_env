import gymnasium as gym
import minihex
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from minihex.interactive.interactive import InteractiveGame
from configparser import ConfigParser
from minihex.HexGame import player as hex_player

from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()

env = selfplay_wrapper(HexEnv)(play_gui=True)
env = ActionMasker(env, mask_fn)
# model.set_env(env)

#model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1) 

#model.set_env(env)
# policy_kwargs = dict(
#     features_extractor_class=CustomPolicy,
#     features_extractor_kwargs=dict(features_dim=128),
# )
# model = MaskablePPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log= "log/") 
model = MaskablePPO.load("best_model_0.17552268213326294")

state, info = env.reset()
terminated = False

while not terminated:
    board = state
    action, _states = model.predict(state, deterministic=True, action_masks=env.legal_actions())
    print("agent played: ", action)
    state, reward, terminated, truncated, info = env.step(action)