import gym
from gym import spaces
import numpy as np
from enum import IntEnum

from minihex.interactive.interactive import InteractiveGame
from configparser import ConfigParser

import torch
from minihex.creation.noise import singh_maddala_onto_output, uniform_noise_onto_output
from minihex.utils import utils
import torch.nn as nn
from torch.distributions.categorical import Categorical
from minihex.utils.utils import load_model

class player(IntEnum):
    BLACK = 0
    WHITE = 1
    EMPTY = 2


class HexGame(object):
    """
    Hex Game Environment.
    """

    def __init__(self, active_player, board,
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
                        self.active_player = player.BLACK
                        self.flood_fill((y, x))
                    elif value == player.WHITE:
                        self.active_player = player.WHITE
                        self.flood_fill((y, x))

        self.active_player = active_player
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
        #     self.winner = (self.active_player + 1) % 2
        #     return (self.active_player + 1) % 2

        y, x = self.action_to_coordinate(action)
        self.board[y, x] = self.active_player
        self.empty_fields -= 1

        self.flood_fill((y, x))

        winner = None
        regions = self.regions[self.active_player]
        if regions[-1, -1] == 1:
            self.done = True
            winner = player(self.active_player)
            self.winner = winner
        elif self.empty_fields <= 0:
            self.done = True
            winner = None

        self.active_player = (self.active_player + 1) % 2
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
        regions = self.regions[self.active_player]

        y, x = (position[0] + 1, position[1] + 1)
        neighborhood = regions[(y - 1):(y + 2), (x - 1):(x + 2)].copy()
        neighborhood[0, 0] = 0
        neighborhood[2, 2] = 0
        adjacent_regions = sorted(set(neighborhood.flatten().tolist()))

        # the region label = 0 is always present, but not a region
        adjacent_regions.pop(0)

        if len(adjacent_regions) == 0:
            regions[y, x] = self.region_counter[self.active_player]
            self.region_counter[self.active_player] += 1
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
                 player_color=player.BLACK,
                 active_player=player.BLACK,
                 board=None,
                 regions=None,
                 board_size=5,
                 debug=False,
                 show_board=False):
        
        self.show_board = show_board

        if board is None:
            board = player.EMPTY * np.ones((board_size, board_size))

        config = ConfigParser()
        config.read('config.ini')

        if self.show_board:
            self.interactive = InteractiveGame(config, board)

        if opponent_policy == "interactive":
            self.opponent_policy = self.interactive_play
        elif opponent_policy == "NN":
            self.opponent_policy = self.batched_single_move
            self.model = load_model(f'minihex/models/{config.get("INTERACTIVE", "model", fallback="11_2w4_2000")}.pt')
            self.temperature=0.1 #config.getfloat("INTERACTIVE", 'temperature', fallback=0.1)
            self.temperature_decay=config.getfloat("INTERACTIVE", 'temperature_decay', fallback=1.)
        else:
            self.opponent_policy = opponent_policy
        
        self.board_size = board_size
        self.initial_board = board
        self.active_player = active_player
        self.player = player_color
        self.simulator = None
        self.winner = None
        self.previous_opponent_move = None
        self.debug = debug
        # cache initial connection matrix (approx +100 games/s)
        self.initial_regions = regions


    @property
    def opponent(self):
        return player((self.player + 1) % 2)

    def reset(self):
        if self.initial_regions is None:
            self.simulator = HexGame(self.active_player,
                                     self.initial_board.copy(),
                                     self.player,
                                     debug=self.debug)
            regions = self.simulator.regions.copy()
            self.initial_regions = regions
        else:
            regions = self.initial_regions.copy()
            self.simulator = HexGame(self.active_player,
                                     self.initial_board.copy(),
                                     self.player,
                                     connected_stones=regions,
                                     debug=self.debug)

        self.previous_opponent_move = None

        if self.player != self.active_player:
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

        return (self.simulator.board, self.active_player), info

    def step(self, action):
        if not self.simulator.done:
            self.winner = self.simulator.make_move(action)
        
        if self.show_board:
            self.interactive.gui.update_board(self.simulator.board)
        
        opponent_action = None

        if not self.simulator.done:
            info_opponent = {
                'state': self.simulator.board,
                'last_move_opponent': action,
                'last_move_player': self.previous_opponent_move
            }
            opponent_action = self.opponent_move(info_opponent)
        if self.show_board:
            self.interactive.gui.update_board(self.simulator.board)

        if self.winner == self.player:
            reward = 1
        elif self.winner == self.opponent:
            reward = -1
        else:
            reward = 0

        info = {
            'state': self.simulator.board,
            'last_move_opponent': opponent_action,
            'last_move_player': action
        }
        # breakpoint()
        return ((self.simulator.board, self.active_player), reward,
                self.simulator.done, info)

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
        opponent_action = self.opponent_policy(self.simulator.board,
                                               self.opponent,
                                               info)
        self.winner = self.simulator.make_move(opponent_action)
        self.previous_opponent_move = opponent_action
        return opponent_action

    def interactive_play(self, board, player,info):
        self.interactive.board = board
        self.interactive.gui.update_board(board)
        action = self.interactive.play_move()
        print(action)
        action = self.simulator.coordinate_to_action(action)
        # self.winner = self.simulator.fast_move(action)
        return action

    ## NN functions
    def batched_single_move(self, board, player,info):      
        self.current_boards = []
        self.current_boards_tensor = torch.Tensor()
        # for board_idx in range(self.batch_size):
        #     if self.boards[board_idx].winner == False:
        #         self.current_boards.append(board_idx)
        # breakpoint()
        # self.current_boards_tensor = torch.Tensor((self.simulator.regions>0).astype(int))
        self.current_boards_tensor = torch.Tensor(np.array((self.simulator.regions>0).astype(int))[np.newaxis])
        self.current_boards.append(self.current_boards_tensor)
        if self.current_boards == []:
            return
        # print(self.current_boards)
        self.current_boards_tensor = self.current_boards_tensor.to(utils.device)

        with torch.no_grad():
            outputs_tensor = self.model(self.current_boards_tensor)

        # if self.noise == 'singh':
        #     noise_alpha, noise_beta, noise_lambda = self.noise_parameters
        #     outputs_tensor = singh_maddala_onto_output(outputs_tensor, noise_alpha, noise_beta, noise_lambda)
        # if self.noise == 'uniform':
        #     noise_probability, = self.noise_parameters
        #     outputs_tensor = uniform_noise_onto_output(outputs_tensor, noise_probability)

        moves_count = 3.0
        # moves_count = len(self.boards[self.current_boards[0]].made_moves)
        positions1d = tempered_moves_selection(outputs_tensor, self.temperature*self.temperature_decay**moves_count)

        # self.output_boards_tensor = torch.cat((self.output_boards_tensor, self.current_boards_tensor.detach().cpu()))
        # self.positions_tensor = torch.cat((self.positions_tensor, positions1d.detach().cpu()))

        # for idx in range(len(self.current_boards)):
        correct_position = utils.correct_position1d(positions1d[0].item(), self.board_size,
            0)
        print(self.simulator.action_to_coordinate(correct_position))
        # self.boards[self.current_boards[idx]].set_stone(correct_position)
        return correct_position

def tempered_moves_selection(output_tensor, temperature):
    #samples with softmax from unnormalized values (if temp>0) and selects move
    if temperature < 10**(-10):
        return output_tensor.argmax(1)
    else:
        temperature_output = output_tensor/temperature
        return Categorical(logits=temperature_output).sample()