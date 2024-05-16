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
import os
import csv

# Random Policy Class
class RandomPolicy:
    def choose_action(self, board, action_mask=None):
        possible_actions = np.where(board.flatten() == 0)[0]  # Only choose from empty positions
        return np.random.choice(possible_actions)

# Custom Callback for Logging
class LoggingCallback(BaseCallback):
    def __init__(self, log_file_path, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_file_path = log_file_path
        self.metrics = []
        self.fieldnames = set()

    def _on_step(self):
        metrics = {
            'ep_len_mean': self.logger.get_log_value('rollout/ep_len_mean'),
            'ep_rew_mean': self.logger.get_log_value('rollout/ep_rew_mean')
        }
        self.metrics.append(metrics)
        return True

    def _on_training_end(self):
        # Save the collected metrics to a CSV file
        with open(self.log_file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(self.fieldnames))
            writer.writeheader()
            for metric in self.metrics:
                writer.writerow(metric)

# Function to create action mask for valid moves
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.legal_actions()

def random_train_and_log_rollout(learning_rate, timesteps, board_size):
    # Generate the log file path with details
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = f"rollout_data_{board_size}x{board_size}_lr_{int(learning_rate*1e5)}_timesteps_{int(timesteps)}_{timestamp}.csv"

    # Create the self-play environment
    env = selfplay_wrapper(HexEnv)(board_size=board_size, buffer_size=1, sample_board=True)
    env = ActionMasker(env, mask_fn)


    # Create the random policy
    random_policy = RandomPolicy()
    env.best_model = random_policy  # Set the random policy as the best model to play against

    # Define the PPO model
    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log="log/", learning_rate=learning_rate)

    # Define the logging callback
    # logging_callback = LoggingCallback(log_file_path)

    # Train the model
    model.learn(total_timesteps=timesteps)
    
    # Save the trained model
    model_name = f"random_{board_size}x{board_size}_lr_{str(learning_rate)}_timesteps_{int(timesteps)}"
    model_path = os.path.join("tournament_dir", model_name)
    model.save(model_path)

    return model

