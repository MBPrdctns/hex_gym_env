import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper
from datetime import datetime
import numpy as np
import random
import csv


class ModelEvaluator:
    def __init__(self, model1, model2, env):
        self.model1 = model1
        self.model2 = model2
        self.env = env

    def play_game(self):
        state = self.env.reset()
        done = False
        current_model = self.model1

        while not done:
            action = current_model.predict(state, deterministic=True)[0]
            state, reward, done, _ = self.env.step(action)
            # Switch turns between model1 and model2
            current_model = self.model2 if current_model == self.model1 else self.model1

        return reward

    def evaluate(self, games):
        model1_wins = 0
        model2_wins = 0

        for game in range(games):
            reward = self.play_game()
            if reward == 1:
                model1_wins += 1
            elif reward == -1:
                model2_wins += 1

        return {
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'draws': draws
        }