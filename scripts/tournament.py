import numpy as np
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from ..minihex import Tournament
from .random_niklas import random_train_and_log_rollout
from .selfplay_niklas import selfplay_train_and_log_rollout
from ..minihex.Tournament import ModelEvaluator

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.legal_actions()


board_size = 11
learning_rates = [1e-3, 1e-4, 1e-5]
timesteps = [1e4, 1e7, 1e10]

def initialize_models(random = True, selfplay=True):
    if random:
        random_models = []
        for timesteps in timesteps:
                for learning_rate in learning_rates:
                    random_models.append(random_train_and_log_rollout(learning_rate=learning_rate, timesteps=timesteps))

    if selfplay:
        selfplay_models = []
        for timesteps in timesteps:
            for learning_rate in learning_rates:
                    selfplay_models.append(selfplay_train_and_log_rollout(learning_rate=learning_rate, timesteps=timesteps))

    return [random_models, selfplay_models]


    



