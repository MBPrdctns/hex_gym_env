"""
models overview:

- hex_selfplay: train first random then selfplay only
- hex_selfplay_eps: train first random then selfplay and with epsilon random, reward short gameplays
- hex_selfplay_eps_reversed: train first random then selfplay and with epsilon random, reward long gameplays
- hex_selfplay_eps_reversed_quick: base hex_selfplay_eps_reversed then selfplay, but every other loop use hex_selfplay_eps and with epsilon random, reward long gameplays
- hex_selfplay_slow_lr_003: train first random then selfplay with decreasing epsilon random, reward {-1, 1}, learning rate 0.003
- hex_selfplay_slow_lr_0015: base hex_selfplay_slow_lr_003 then selfplay with decreasing epsilon random, reward {-1, 1}, learning rate 0.0015

Remarks:
- learning_rate default: 0.0003
"""




import minihex
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableActorCriticCnnPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
from functools import partial
#from minihex.HexSingleGame import player as hex_player
import random
from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper
from minihex.EvaluationCallback import SelfPlayCallback


from minihex.models.customPolicies import CustomPolicy

N = 1e7

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.legal_actions()

CustomPolicy = CustomPolicy
# model = MaskablePPO.load("hex_selfplay_new")

env = selfplay_wrapper(HexEnv)() #(base_model=model)
env = ActionMasker(env, mask_fn)
# model.set_env(env)

# model = MaskablePPO(CustomPolicy, env, verbose=1)
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1) 

callback = SelfPlayCallback(eval_env=env, gym_env=env, eval_freq = 100000) #  eval_freq defaul 10000
model.learn(N, callback=[callback])
# model.save("hex_selfplay_new_nn")