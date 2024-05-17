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
import random
from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper
from minihex.EvaluationCallback import SelfPlayCallback

MODEL_NAME = "8x8_MLP-default_lr-0.0003"
LEARNING_RATE = 0.0003

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.legal_actions()

env = selfplay_wrapper(HexEnv)(board_size=8,
                                buffer_size=30,
                              scores=np.zeros(30),
                              sample_board = False)
env = ActionMasker(env, mask_fn)
policy_kwargs = None
model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log= "log/", policy_kwargs=policy_kwargs, learning_rate=LEARNING_RATE) 

callback = SelfPlayCallback(eval_env=env, 
                            gym_env=env, 
                            eval_freq=10000, 
                            n_eval_episodes=30,
                            model_log_name=MODEL_NAME) 
model.learn(1e9, callback=[callback], tb_log_name=MODEL_NAME)