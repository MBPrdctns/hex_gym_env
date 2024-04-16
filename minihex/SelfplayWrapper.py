import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum
import random
from minihex.__init__ import random_policy 
from minihex.interactive.interactive import InteractiveGame
from configparser import ConfigParser

class BaseRandomPolicy(object):
    def choose_action(self, board, action_mask = None):
        actions = np.arange(board.shape[0] * board.shape[1])
        valid_actions = actions[board.flatten() == 0]
        choice = int(random.random() * len(valid_actions))
        action = valid_actions[choice]
        return action

class OpponentPolicy(object):
    def __init__(self, model):
        self.opponent_model = model

    def choose_action(self, board, action_mask = None):
        action, _ = self.opponent_model.predict(board, deterministic=True, action_masks=action_mask)
        return action
    
def selfplay_wrapper(env):
    class SelfPlayEnv(env):
        def __init__(self, base_model=BaseRandomPolicy()):
            super(SelfPlayEnv, self).__init__()

            if base_model != BaseRandomPolicy:
                base_model = OpponentPolicy(base_model)

            self.opponent_models = [base_model]
            self.best_model = [base_model]
            self.best_mean_reward = -np.inf

        def reset(self, seed=None, options=None):
            super(SelfPlayEnv, self).reset()
            self.agent_player_num = random.randint(0,1)
            self.setup_opponents()

            if self.current_player_num != self.agent_player_num:   
                self.continue_game()

            info = {
                'state': self.simulator.board,
                'last_move_opponent': None, # self.previous_opponent_move,
                'last_move_player': None
            }

            return self.simulator.board, info
    
        def setup_opponents(self):
            rv = random.uniform(0,1)
            if rv < 0.8:
                i = int(random.random() * len(self.best_model))
                self.opponent_model = self.best_model[i]
            else:
                i = int(random.random() * len(self.opponent_models))
                self.opponent_model = self.opponent_models[i]

        def append_opponent_model(self, opponent_model, best_model = False, mean_reward = None):
            new_opponent = OpponentPolicy(opponent_model)
            if best_model:
                self.best_model.append(new_opponent)
                self.best_mean_reward = mean_reward
            
            self.opponent_models.append(new_opponent)
        
        def get_best_mean_reward(self):
            return self.best_mean_reward

        def continue_game(self):
            observation = None 
            reward = None
            done = None

            # self.render()
            action = self.opponent_model.choose_action(self.simulator.board, self.legal_actions())
            observation, reward, done, _ = super(SelfPlayEnv, self).step(action)

            return observation, reward, done, None

        def step(self, action):
            # self.render()
            observation, reward, done, _ = super(SelfPlayEnv, self).step(action)

            if not done:
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward, done, _ = package

            agent_reward = reward[self.agent_player_num]

            # if done:
                # self.render()

            return observation, agent_reward, done, False, {} 
    return SelfPlayEnv