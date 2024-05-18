import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum
import random
from minihex.__init__ import random_policy 
from minihex.interactive.interactive import InteractiveGame
from configparser import ConfigParser

player = {
    "BLACK": {"id": 0, "board_encoding": -1}, # ROT
    "WHITE": {"id": 1, "board_encoding": 1},
    "EMPTY": {"id": 2, "board_encoding": 0}
}

class BaseRandomPolicy(object):
    def choose_action(self, board, action_mask = None):
        actions = np.arange(board.shape[0] * board.shape[1])
        valid_actions = actions[board.flatten() == 0]
        choice = int(random.random() * len(valid_actions))
        action = valid_actions[choice]
        return action
    def save_model(self, path):
        return None

class OpponentPolicy(object):
    def __init__(self, model):
        self.opponent_model = model

    def choose_action(self, board, action_mask = None):
        action, _ = self.opponent_model.predict(board, deterministic=False, action_masks=action_mask)
        return action
    
    def save_model(self, path):
        self.opponent_model.save(path)

def selfplay_wrapper(env):
    class SelfPlayEnv(env):
        def __init__(self, base_model=BaseRandomPolicy(), scores=np.zeros(20), play_gui=False,board_size=5,buffer_size=20, sample_board=False):
            super(SelfPlayEnv, self).__init__(board_size=board_size, sample_board = sample_board)

            if play_gui:
                self.opponent_models = [InteractiveGame(self.initial_board)]
                self.opponent_scores = [1]
                base_model = InteractiveGame(self.initial_board)
            else:
                if type(base_model) != BaseRandomPolicy:
                    self.opponent_models = np.array([OpponentPolicy(base_model) for _ in range(buffer_size)])
                    self.opponent_scores = scores
                    base_model = OpponentPolicy(base_model)
                else:
                    self.opponent_models = np.array([BaseRandomPolicy() for _ in range(buffer_size)])
                    self.opponent_scores = np.zeros(buffer_size) #np.array(range(20))

            self.best_model = base_model
            self.best_score = np.max(self.opponent_scores)
            self.best_mean_reward = -np.inf
            self.eval_state = False
            self.eval_episode = 0
            self.play_gui = play_gui

        def reset(self, seed=None, options=None):
            super(SelfPlayEnv, self).reset()
            self.agent_player_num = random.randint(0,1)
            self.setup_opponents()
            if self.play_gui:
                self.best_model.gui.update_board(self.simulator.board)

            if self.current_player_num != self.agent_player_num:   
                self.continue_game()
                # self.invert_board() # if genau dann, wenn gegner (buffer) schwarz ist -> model braucht inverted board.

            info = {
                'state': self.simulator.board,
                'last_move_opponent': None, # self.previous_opponent_move,
                'last_move_player': None
            }

            return self.simulator.board, info
    
        def setup_opponents(self):
            if self.eval_state:
                if self.eval_episode <= 29:
                    self.opponent_model = self.opponent_models[self.eval_episode]
                    self.eval_episode += 1
                return None
            rv = random.uniform(0,1)
            if rv < 0.8:
                # i = int(random.random() * len(self.best_model))
                # self.opponent_model = self.best_model[i]
                self.opponent_model = self.best_model
            else:
                i = int(random.random() * len(self.opponent_models))
                self.opponent_model = self.opponent_models[i]

        def append_opponent_model(self, opponent_model, best_model = False, mean_reward = None):
            new_opponent = OpponentPolicy(opponent_model)
            if best_model:
                self.best_model = new_opponent
                self.best_mean_reward = mean_reward
            
            self.opponent_models.append(new_opponent)
        
        def get_best_mean_reward(self):
            return self.best_mean_reward

        def set_eval(self, eval_state):
            self.eval_episode = 0
            self.eval_state = eval_state
            assert(len(self.opponent_models)==len(self.opponent_scores))

        def get_scores(self):
            return self.opponent_scores
        
        def set_opponent_model(self, index, model, score):
            print("Previous model scores: ", self.opponent_scores)
            model = OpponentPolicy(model)
            self.opponent_models[index] = model
            #self.opponent_scores += 1
            self.opponent_scores[index] = score
            print("Replaced model from index: ", index)
            print("Model score: ", score)
            if score > self.best_score:
                self.best_model = model
                self.best_score = score
                print("NEW BEST MODEL WITH SCORE: ", score)
                
        
        def get_opponent_models(self):
            return self.opponent_models
        
        def save_best_model(self):
            model_name = "models/best_model_" + str(self.best_score)
            self.best_model.save_model(model_name)

        def continue_game(self):
            observation = None 
            reward = None
            done = None

            if self.play_gui & (self.current_player_num == player["WHITE"]["id"]): 
                self.invert_board()

            rv = random.uniform(0,1)
            if rv > 0.05:
                action = self.opponent_model.choose_action(self.simulator.board, self.legal_actions())
            else:
                action = BaseRandomPolicy().choose_action(self.simulator.board)

            if self.play_gui & (self.current_player_num == player["WHITE"]["id"]): 
                self.invert_board()
                x,y = self.simulator.action_to_coordinate(action)
                action = self.simulator.coordinate_to_action((y,x))

            observation, reward, done, _ = super(SelfPlayEnv, self).step(action)

            return observation, reward, done, None

        def step(self, action):
            # self.render()
            observation, reward, done, _ = super(SelfPlayEnv, self).step(action)

            if self.play_gui:
                if self.current_player_num == player["WHITE"]["id"]: # already added +1 to current_player_num in step
                    self.invert_board()
                self.opponent_model.gui.update_board(self.simulator.board)
                self.get_probs(self.opponent_model.model)

                if self.current_player_num == player["WHITE"]["id"]:
                    self.invert_board()
                

            if not done:
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward, done, _ = package

            agent_reward = reward[self.agent_player_num]
            # print("agent_reward: ", agent_reward)
            # print(self.agent_player_num)

            # if done:
                # self.render()

            return observation, agent_reward, done, False, {} 
        
        def get_probs(self, model):
            board_tensor = model.policy.obs_to_tensor(self.simulator.board)
            probs = model.policy.get_distribution(board_tensor[0]).distribution.probs
            probs_np = probs.detach().numpy()[0]
            scaled_probs = scale_value(probs_np, probs_np[self.simulator.board.flatten()==0].min(), probs_np[self.simulator.board.flatten()==0].max())
            self.opponent_model.gui.update_field_text(np.round(scaled_probs, 1), self.simulator.board)

    return SelfPlayEnv

def scale_value(value, min_val, max_val):
    return 10 * (value - min_val) / (max_val - min_val)