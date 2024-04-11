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
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
from functools import partial
#from minihex.HexSingleGame import player as hex_player
import random
from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper
from minihex.EvaluationCallback import SelfPlayCallback

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.legal_actions()

# opponent_policy = minihex.random_policy
# player = hex_player.BLACK
# model = MaskablePPO.load("hex_selfplay_slow_lr_003")
# model.learning_rate = 0.0015 ## overwrite

env = selfplay_wrapper(HexEnv)()
env = ActionMasker(env, mask_fn)

model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1) 

callback = SelfPlayCallback(eval_env=env)
model.learn(100000, callback=[callback])
model.save("hex_selfplay_new")

# for i in range(1000):
#     print("Iteration: ", i)
    
#     model.learn(total_timesteps=500, log_interval=4)
#     # model.save("hex_selfplay_slow_lr_0015")
#     # model.learning_rate = 0.003 - i * 0.000027 ## overwrite lr

#     if not i % 10:
#         model_history.append(model)

#     opponent_model = random.choice(model_history)

#     # env.set_opponent_model(model)
#     # opponent_policy = env.opponent_predict #, deterministic=True, action_masks=env.get_action_mask())

#     env = gym.make("hex-v0",
#                 opponent_model=opponent_model,
#                 opponent_policy="opponent_predict",
#                 player_color=player,
#                 board_size=5,
#                 eps = 0
#                 )
#     env = ActionMasker(env, mask_fn)
#     model.set_env(env)
#     state, info = env.reset()
# model.save("hex_selfplay_shuffled_history")
# #del model # remove to demonstrate saving and loading
