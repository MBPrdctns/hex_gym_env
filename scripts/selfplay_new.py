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
from minihex.CustomNetwork import CustomPolicy
#from minihex.models.pytorch_NN import CustomPolicy

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.legal_actions()

#model = MaskablePPO.load("hex_selfplay_new")

env = selfplay_wrapper(HexEnv)()#base_model=model)
env = ActionMasker(env, mask_fn)
# model.set_env(env)

#model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1) 

#model.set_env(env)
policy_kwargs = dict(
    features_extractor_class=CustomPolicy,
    features_extractor_kwargs=dict(features_dim=128),
)
model = MaskablePPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs) 

callback = SelfPlayCallback(eval_env=env, gym_env=env, eval_freq=10000) #  eval_freq defaul 10000
model.learn(5e6, callback=[callback])
model.save("hex_selfplay_new")