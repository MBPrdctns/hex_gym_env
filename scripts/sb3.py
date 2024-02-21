import minihex
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("hex-v0",
               opponent_policy=minihex.random_policy,
               board_size=11)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("hex")

del model # remove to demonstrate saving and loading

model = DQN.load("hex")

obs, info = env.reset()

"""
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
"""