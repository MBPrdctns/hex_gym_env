import minihex
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableActorCriticCnnPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
from functools import partial
#from minihex.HexSingleGame import player as hex_player
import os
from minihex.HexSingleGame import HexEnv
from minihex.SelfplayWrapper import selfplay_wrapper
from minihex.EvaluationCallback import SelfPlayCallback
from minihex.CustomNetwork import CustomPolicy
#from minihex.models.pytorch_NN import CustomPolicy

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.legal_actions()

def selfplay_train_and_log_rollout(learning_rate, timesteps, board_size):
    env = selfplay_wrapper(HexEnv)(board_size=board_size,
                                    buffer_size=20,
                                    scores=np.zeros(20),
                                    sample_board = True)

    env = ActionMasker(env, mask_fn)

    policy_kwargs = None

    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log= "log/", learning_rate = learning_rate) 

    callback = SelfPlayCallback(eval_env=env, 
                                gym_env=env, 
                                eval_freq=1000, 
                                n_eval_episodes=20) #  eval_freq defaul 10000
    model.learn(timesteps, callback=[callback])

    # save model to directory
    model_name = f"selfplay_{board_size}x{board_size}_lr_{str(learning_rate)}_timesteps_{int(timesteps)}"
    model_path = os.path.join("tournament_dir", model_name)
    model.save(model_path)

    return model