import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum
import random
from minihex.__init__ import random_policy 
from minihex.interactive.interactive import InteractiveGame
from configparser import ConfigParser

class player(IntEnum):
    BLACK = 0
    WHITE = 1
    EMPTY = 2


class HexGame(object):
    """
    Hex Game Environment.
    """

    def __init__(self, current_player_num, board,
                 focus_player, connected_stones=None, debug=False):
        self.board = board
        # track number of empty feelds for speed
        self.empty_fields = np.count_nonzero(board == player.EMPTY)

        if debug:
            self.make_move = self.make_move_debug
        else:
            self.make_move = self.fast_move

        # self.special_moves = IntEnum("SpecialMoves", {
        #     "RESIGN": self.board_size ** 2,
        #     "SWAP": self.board_size ** 2 + 1
        # })

        if connected_stones is None:
            self.regions = np.stack([
                np.pad(np.zeros_like(self.board), 1),
                np.pad(np.zeros_like(self.board), 1)
            ], axis=0)
            self.regions[player.WHITE][:, 0] = 1
            self.regions[player.BLACK][0, :] = 1
            self.regions[player.WHITE][:, self.board_size + 1] = 2
            self.regions[player.BLACK][self.board_size + 1, :] = 2
        else:
            self.regions = connected_stones

        self.region_counter = np.zeros(2)
        self.region_counter[player.BLACK] = np.max(self.regions[player.BLACK]) + 1
        self.region_counter[player.WHITE] = np.max(self.regions[player.WHITE]) + 1

        if connected_stones is None:
            for y, row in enumerate(board):
                for x, value in enumerate(row):
                    if value == player.BLACK:
                        self.current_player_num = player.BLACK
                        self.flood_fill((y, x))
                    elif value == player.WHITE:
                        self.current_player_num = player.WHITE
                        self.flood_fill((y, x))

        self.current_player_num = current_player_num
        self.player = focus_player
        self.done = False
        self.winner = None

        self.actions = np.arange(self.board_size ** 2)

    @property
    def board_size(self):
        return self.board.shape[1]

    def is_valid_move(self, action):
        coords = self.action_to_coordinate(action)
        return self.board[coords[0], coords[1]] == player.EMPTY

    def make_move_debug(self, action):
        if not self.is_valid_move(action):
            raise IndexError(("Illegal move "
                             f"{self.action_to_coordinate(action)}"))

        return self.fast_move(action)

    def fast_move(self, action):
        # # currently resigning is not a possible option
        # if action == self.special_moves.RESIGN:
        #     self.done = True
        #     self.winner = (self.current_player_num + 1) % 2
        #     return (self.current_player_num + 1) % 2
        if not self.is_valid_move(action):
            return 3
        
        y, x = self.action_to_coordinate(action)
        self.board[y, x] = self.current_player_num
        self.empty_fields -= x

        self.flood_fill((y, x))

        winner = None
        regions = self.regions[self.current_player_num]
        if regions[-1, -1] == 1:
            self.done = True
            winner = player(self.current_player_num)
            self.winner = winner
        # elif self.empty_fields <= 0:
        #     self.done = True
        #     winner = None

        self.current_player_num = (self.current_player_num + 1) % 2
        return winner

    def coordinate_to_action(self, coords):
        return np.ravel_multi_index(coords, (self.board_size, self.board_size))

    def action_to_coordinate(self, action):
        y = action // self.board_size
        x = action - self.board_size * y
        return (y, x)

    def get_possible_actions(self):
        return self.actions[self.board.flatten() == player.EMPTY]

    def flood_fill(self, position):
        regions = self.regions[self.current_player_num]
        y, x = (position[0] + 1, position[1] + 1)
        neighborhood = regions[(y - 1):(y + 2), (x - 1):(x + 2)].copy()
        neighborhood[0, 0] = 0
        neighborhood[2, 2] = 0
        adjacent_regions = sorted(set(neighborhood.flatten().tolist()))

        # the region label = 0 is always present, but not a region
        adjacent_regions.pop(0)

        if len(adjacent_regions) == 0:
            regions[y, x] = self.region_counter[self.current_player_num]
            self.region_counter[self.current_player_num] += 1
        else:
            new_region_label = adjacent_regions.pop(0)
            regions[y, x] = new_region_label
            for label in adjacent_regions:
                regions[regions == label] = new_region_label


class HexEnv(gym.Env):
    """
    Hex environment. Play against a fixed opponent.
    """

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, opponent_policy,
                 opponent_model=None,
                 player_color=player.BLACK,
                 current_player_num=player.BLACK,
                 board=None,
                 regions=None,
                 board_size=5,
                 debug=False,
                 show_board=False,
                 eps=0.5):
        
        if opponent_policy == "interactive":
            self.opponent_policy = self.interactive_play
            self.interactive = True
        elif opponent_policy == "opponent_predict":
            self.opponent_policy = self.opponent_predict
            self.interactive=False
        else:
            self.opponent_policy = opponent_policy
            self.interactive = False
        
        if board is None:
            board = player.EMPTY * np.ones((board_size, board_size))
        
        self.n_players = 2 # SIMPLE setup

        self.eps = eps
        self.opponent_model = opponent_model
        self.initial_board = board
        self.current_player_num = current_player_num
        self.player = player_color
        self.simulator = None
        self.winner = None
        self.previous_opponent_move = None
        self.debug = debug
        self.show_board = show_board        
        self.board_size = board_size
        self.observation_space = spaces.Box(low=0, high=2, shape=(board_size, board_size), dtype=np.uint8)
        self.action_space = spaces.Discrete(board_size**2)
        # cache initial connection matrix (approx +100 games/s)
        self.initial_regions = regions

        if self.show_board:
            config = ConfigParser()
            config.read('config.ini')
            self.interactive = InteractiveGame(config, board)

    @property
    def opponent(self):
        return player((self.player + 1) % 2)

    def get_action_mask(self):
        return np.array([self.simulator.is_valid_move(action) for action in range(self.board_size**2)])

    def reset(self, seed=None, options=None):
        if self.initial_regions is None:
            self.simulator = HexGame(self.current_player_num,
                                     self.initial_board.copy(),
                                     self.player,
                                     debug=self.debug)
            regions = self.simulator.regions.copy()
            self.initial_regions = regions
        else:
            regions = self.initial_regions.copy()
            self.simulator = HexGame(self.current_player_num,
                                     self.initial_board.copy(),
                                     self.player,
                                     connected_stones=regions,
                                     debug=self.debug)

        self.previous_opponent_move = None

        if self.player != self.current_player_num:
            info_opponent = {
                'state': self.simulator.board,
                'last_move_opponent': None,
                'last_move_player': None
            }
            self.opponent_move(info_opponent)

        info = {
            'state': self.simulator.board,
            'last_move_opponent': self.previous_opponent_move,
            'last_move_player': None
        }
        # current_player_num_array = np.full((self.board_size, self.board_size), self.current_player_num)
        # state = np.stack([self.simulator.board, current_player_num_array], axis=0)
        # print(state)
        # print(self.observation_space)

        return self.simulator.board, info

    def step(self, action):
        if self.player != player.BLACK:
            self.invert_board()
            y, x = self.simulator.action_to_coordinate(action)
            action = self.simulator.coordinate_to_action((x, y))
            
        if not self.simulator.done:
            self.winner = self.simulator.make_move(action)
            if self.winner == 3: # invalid move
                self.simulator.done = True

        

        opponent_action = None

        if not self.simulator.done:
            info_opponent = {
                'state': self.simulator.board,
                'last_move_opponent': action,
                'last_move_player': self.previous_opponent_move
            }
            opponent_action = self.opponent_move(info_opponent)

        if self.winner == self.player:
            reward = 1
            # reward = (self.simulator.board_size ** 2 - 1) - (self.simulator.board == self.player).sum()
            # reward = (self.simulator.board == self.player).sum()
            # print(reward)
        elif self.winner == self.opponent:
            # reward = - 12 # (self.simulator.board == self.player).sum()
            reward = -1
            # print(reward)
        elif self.winner == 3: # invalid move
            reward = -100
        else:
            reward = 0

        info = {
            'state': self.simulator.board,
            'last_move_opponent': opponent_action,
            'last_move_player': action,
            'winner': self.winner
        }

        # current_player_num_array = np.full((self.board_size, self.board_size), self.current_player_num)
        # state = np.stack([self.simulator.board, current_player_num_array], axis=0)

        if self.player != player.BLACK:
            self.invert_board()

        return (self.simulator.board, reward,
                self.simulator.done, False, info)
    
    def invert_board(self):
        board = self.simulator.board.copy()
        inverted_board = board.T
        inverted_board[inverted_board==player.BLACK] = -1 # placeholder
        inverted_board[inverted_board==player.WHITE] = player.BLACK
        inverted_board[inverted_board == -1] = player.WHITE
        self.simulator.board = inverted_board

    def render(self, mode='ansi', close=False):
        
        board = self.simulator.board
        print(" " * 6, end="")
        for j in range(board.shape[1]):
            print(" ", j + 1, " ", end="")
            print("|", end="")
        print("")
        print(" " * 5, end="")
        print("-" * (board.shape[1] * 6 - 1), end="")
        print("")
        for i in range(board.shape[1]):
            print(" " * (1 + i * 3), i + 1, " ", end="")
            print("|", end="")
            for j in range(board.shape[1]):
                if board[i, j] == player.EMPTY:
                    print("  O  ", end="")
                elif board[i, j] == player.BLACK:
                    print("  B  ", end="")
                else:
                    print("  W  ", end="")
                print("|", end="")
            print("")
            print(" " * (i * 3 + 1), end="")
            print("-" * (board.shape[1] * 7 - 1), end="")
            print("")

    def opponent_move(self, info):
        if (self.player == player.BLACK and not self.interactive): # if opponent plays black, invert board - model gets state always playing black
            self.invert_board()
            # print(self.simulator.board)
        opponent_action = self.opponent_policy(self.simulator.board)
        # print(opponent_action)
        if self.player == player.BLACK and not self.interactive:
            self.invert_board()
            # print(self.simulator.board)
            y, x = self.simulator.action_to_coordinate(opponent_action)
            opponent_action = self.simulator.coordinate_to_action((x, y))
            # print(opponent_action)

        # 1
        self.winner = self.simulator.make_move(opponent_action)
        self.previous_opponent_move = opponent_action
            
        return opponent_action
    
    def set_opponent_model(self, model):
        self.opponent_model = model
    
    def opponent_predict(self, state):
        rv = random.uniform(0,1)
        if rv < self.eps:
            return random_policy(state)
        action, _ = self.opponent_model.predict(state, deterministic=True, action_masks=self.get_action_mask())
        return action

    def interactive_play(self, board):
        self.interactive.board = board
        # self.interactive.gui.update_board(board)
        action = self.interactive.play_move()
        print(action)
        action = self.simulator.coordinate_to_action(action)
        # self.winner = self.simulator.fast_move(action)
        return action
    
    def legal_actions(self):
        return (self.interactive.board == 2)