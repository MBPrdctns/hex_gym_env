"""
models overview:

- hex_selfplay: train first random then selfplay only
- hex_selfplay_eps: train first random then selfplay and with epsilon random, reward short gameplays
- hex_selfplay_eps_reversed: train first random then selfplay and with epsilon random, reward long gameplays
- hex_selfplay_eps_reversed_quick: base hex_selfplay_eps_reversed then selfplay, but every other loop use hex_selfplay_eps and with epsilon random, reward long gameplays
- hex_selfplay_slow_lr_003: train first random then selfplay with decreasing epsilon random, reward {-1, 1}, learning rate 0.003
- hex_selfplay_slow_lr_0015: base hex_selfplay_slow_lr_003 then selfplay with decreasing epsilon random, reward {-1, 1}, learning rate 0.0015
- hex_selfplay

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
from minihex.HexGame import player as hex_player
import random

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()

opponent_policy = minihex.random_policy
player = hex_player.BLACK
model = MaskablePPO.load("hex_selfplay_shuffled_history")
model.learning_rate = 0.001 ## overwrite

env = gym.make("hex-v0",
                # opponent_model=model,
                # opponent_policy = "opponent_predict",
                opponent_policy = opponent_policy,
                player_color=player,
                board_size=5,
                eps=0)
env = ActionMasker(env, mask_fn)

# model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1) 
model.set_env(env)

#env = gym.make("hex-v0",
            #    opponent_policy=opponent_policy,
            #    player_color=player,
            #    board_size=5)
#env = ActionMasker(env, mask_fn)
# model = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=0.0015, verbose=1) 
state, info = env.reset()

model_history = []

for i in range(2000):
    print("Iteration: ", i)
    
    model.learn(total_timesteps=500, log_interval=4)
    # model.save("hex_selfplay_slow_lr_0015")
    # model.learning_rate = 0.003 - i * 0.000027 ## overwrite lr

    if not i % 10:
        model_history.append(model)

    if i % 2:
        current_player_num = hex_player.WHITE
    else:
        current_player_num = hex_player.BLACK

    opponent_model = random.choice(model_history)

    # env.set_opponent_model(model)
    # opponent_policy = env.opponent_predict #, deterministic=True, action_masks=env.get_action_mask())

    env = gym.make("hex-v0",
                opponent_model=opponent_model,
                opponent_policy="opponent_predict",
                player_color=player,
                current_player_num=current_player_num,
                board_size=5,
                eps = 0 #.3 * (1 - i/100)
                )
    env = ActionMasker(env, mask_fn)
    model.set_env(env)
    state, info = env.reset()
model.save("hex_selfplay_shuffled_history")
#del model # remove to demonstrate saving and loading
