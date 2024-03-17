import minihex
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
from functools import partial
from minihex.HexGame import player as hex_player
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()

#opponent_policy = minihex.random_policy
player = hex_player.BLACK
model = MaskablePPO.load("hex_selfplay")
env = gym.make("hex-v0",
                opponent_model=model,
               opponent_policy="opponent_predict",
               player_color=player,
               board_size=5)
env = ActionMasker(env, mask_fn)
model.set_env(env)

#env = gym.make("hex-v0",
            #    opponent_policy=opponent_policy,
            #    player_color=player,
            #    board_size=5)
#env = ActionMasker(env, mask_fn)
#model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1) 
state, info = env.reset()

for i in range(100):
    print("Iteration: ", i)
    
    model.learn(total_timesteps=40000, log_interval=4)
    model.save("hex_selfplay")

    # env.set_opponent_model(model)
    # opponent_policy = env.opponent_predict #, deterministic=True, action_masks=env.get_action_mask())

    env = gym.make("hex-v0",
                opponent_model=model,
               opponent_policy="opponent_predict",
               player_color=player,
               board_size=5)
    env = ActionMasker(env, mask_fn)
    model.set_env(env)
    state, info = env.reset()

#del model # remove to demonstrate saving and loading


env = gym.make("hex-v0",
               opponent_policy=opponent_policy,
               player_color=player,
               board_size=5)
env = ActionMasker(env, mask_fn)
model = MaskablePPO.load("hex_selfplay")
obs, info = env.reset()

episodes = 100
winners = []
for ep in range (episodes):
    obs, info = env.reset()
    terminated = False
    while not terminated:
        # env.render()
        action, _states = model.predict(obs, deterministic=True, action_masks=env.get_action_mask())
        obs, reward, terminated, truncated, info = env.step(action)
    winners.append(info["winner"])
    print(info["winner"], "won!")

    #env.render()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         print(info["state"])
#         obs, info = env.reset()
