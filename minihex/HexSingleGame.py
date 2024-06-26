import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum
import random
from minihex.__init__ import random_policy 
from minihex.interactive.interactive import InteractiveGame
from configparser import ConfigParser

# class player(IntEnum):
#     BLACK = -1
#     WHITE = 1
#     EMPTY = 0

player = {
    "BLACK": {"id": 0, "board_encoding": -1}, # ROT
    "WHITE": {"id": 1, "board_encoding": 1},
    "EMPTY": {"id": 2, "board_encoding": 0}
}

class HexGame(object):
    """
    Hex Game Environment.
    """

    def __init__(self, active_player, board, connected_stones=None, debug=False):
        self.board = board
        # track number of empty feelds for speed
        self.empty_fields = np.count_nonzero(board == player["EMPTY"]["board_encoding"])

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
            self.regions[player["WHITE"]["id"]][:, 0] = 1
            self.regions[player["BLACK"]["id"]][0, :] = 1
            self.regions[player["WHITE"]["id"]][:, self.board_size + 1] = 2
            self.regions[player["BLACK"]["id"]][self.board_size + 1, :] = 2
        else:
            self.regions = connected_stones

        self.region_counter = np.zeros(2)
        self.region_counter[player["BLACK"]["id"]] = np.max(self.regions[player["BLACK"]["id"]]) + 1
        self.region_counter[player["WHITE"]["id"]] = np.max(self.regions[player["WHITE"]["id"]]) + 1

        if connected_stones is None:
            for y, row in enumerate(board):
                for x, value in enumerate(row):
                    if value == player["BLACK"]["board_encoding"]:
                        self.current_player_num  = player["BLACK"]["id"]
                        self.flood_fill((y, x))
                    elif value == player["WHITE"]["board_encoding"]:
                        self.current_player_num  = player["WHITE"]["id"]
                        self.flood_fill((y, x))

        self.current_player_num  = active_player
        self.done = False
        self.winner = None

        self.actions = np.arange(self.board_size ** 2)

    @property
    def board_size(self):
        return self.board.shape[1]

    def is_valid_move(self, action):
        coords = self.action_to_coordinate(action)
        return self.board[coords[0], coords[1]] == player["EMPTY"]["board_encoding"]

    def make_move_debug(self, action):
        if not self.is_valid_move(action):
            raise IndexError(("Illegal move "
                             f"{self.action_to_coordinate(action)}"))

        return self.fast_move(action)

    def fast_move(self, action):
        # # currently resigning is not a possible option
        # if action == self.special_moves.RESIGN:
        #     self.done = True
        #     self.winner = (self.current_player_num  + 1) % 2
        #     return (self.current_player_num  + 1) % 2
        # print("current player fast move:", self.current_player_num)
        if not self.is_valid_move(action):
            return 3
        
        y, x = self.action_to_coordinate(action)
        # self.board[y, x] = self.current_player_num 
        self.board[y, x] = player["BLACK"]["board_encoding"]# if self.current_player_num == 0 else player["WHITE"]["board_encoding"]
        self.empty_fields -= 1
        # print(self.board)
        if self.current_player_num == player["WHITE"]["id"]:
            self.flood_fill((x, y)) # switch
        else:
            self.flood_fill((y, x))


        winner = None
        regions = self.regions[self.current_player_num]
        if regions[-1, -1] == 1:
            self.done = True
            winner = self.current_player_num
            # winner = player(self.current_player_num)
            self.winner = winner
            # print(winner)
        elif self.empty_fields <= 0:
            self.done = True
            winner = None
        # print(self.board)
        self.current_player_num = (self.current_player_num + 1) % 2
        return winner

    def coordinate_to_action(self, coords):
        return np.ravel_multi_index(coords, (self.board_size, self.board_size))

    def action_to_coordinate(self, action):
        y = action // self.board_size
        x = action - self.board_size * y
        return (y, x)

    def get_possible_actions(self):
        return self.actions[self.board.flatten() == player["EMPTY"]["board_encoding"]]

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
            regions[y, x] = self.region_counter[self.current_player_num ]
            self.region_counter[self.current_player_num ] += 1
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

    def __init__(self,
                 # player_color=player.BLACK,
                 current_player_num=player["BLACK"]["id"],
                 board=None,
                 regions=None,
                 board_size=5,
                 debug=False,
                 show_board=False,
                 eps=0.5,
                 sample_board = False):
        
        if board is None and sample_board == False:
            board = player["EMPTY"]["board_encoding"] * np.ones((board_size, board_size))
        elif sample_board:
            board = self.random_board(player["EMPTY"]["board_encoding"] * np.ones((board_size, board_size)))

        self.sample_board = sample_board
        self.eps = eps
        self.initial_board = board
        self.current_player_num = current_player_num
        self.simulator = None
        self.winner = None
        self.debug = debug
        self.show_board = show_board        
        self.board_size = board_size
        self.observation_space = spaces.Box(low=-1, high=1, shape=(board_size, board_size), dtype=int)
        self.action_space = spaces.Discrete(board_size**2)
        # cache initial connection matrix (approx +100 games/s)
        self.initial_regions = regions

        if self.show_board:
            config = ConfigParser()
            config.read('config.ini')
            self.interactive = InteractiveGame(config, board)

    # def get_action_mask(self):
    #    return np.array([self.simulator.is_valid_move(action) for action in range(self.board_size**2)])

    @property
    def observation(self):
        return self.simulator.board

    def legal_actions(self):
        return np.array([self.simulator.is_valid_move(action) for action in range(self.board_size**2)])

    def reset(self, seed=None, options=None):
        self.current_player_num = player["BLACK"]["id"]

        if self.initial_regions is None and self.sample_board == False:
            self.simulator = HexGame(self.current_player_num,
                                     self.initial_board.copy(),
                                     debug=self.debug)
            regions = self.simulator.regions.copy()
            self.initial_regions = regions
        elif self.sample_board:
            self.simulator = HexGame(self.current_player_num,
                                     self.random_board(player["EMPTY"]["board_encoding"] * np.ones((self.board_size, self.board_size))),
                                     debug=self.debug)
            regions = self.simulator.regions.copy()
            self.initial_regions = regions
        else:
            regions = self.initial_regions.copy()
            self.simulator = HexGame(self.current_player_num,
                                     self.initial_board.copy(),
                                     connected_stones=regions,
                                     debug=self.debug)


        return self.simulator.board

    def step(self, action):
        # if self.current_player_num != player["BLACK"]["id"]:
        #     self.invert_board()
        #     y, x = self.simulator.action_to_coordinate(action)
        #     action = self.simulator.coordinate_to_action((x, y))
            
        self.winner = self.simulator.make_move(action)
        if self.winner == 3: # invalid move
            self.simulator.done = True
            reward = -100


        if self.winner == self.current_player_num:
            r = 1 #self.simulator.empty_fields
        elif self.winner == (self.current_player_num + 1) % 2:
            r = -1 #0
        else:
            r = 0

        reward = [-r,-r]
        reward[self.current_player_num] = r
        # reward[(self.current_player_num + 1) % 2] = - 25 + r

        # if self.current_player_num != player["BLACK"]["id"]:
        #     self.invert_board()

        self.current_player_num = (self.current_player_num + 1 )% 2 
        self.invert_board()

        return (self.simulator.board, reward,
                self.simulator.done, {})
    
    def invert_board(self):
        board = self.simulator.board.copy()
        inverted_board = board.T
        inverted_board[inverted_board==player["BLACK"]["board_encoding"]] = -2 # placeholder
        inverted_board[inverted_board==player["WHITE"]["board_encoding"]] = player["BLACK"]["board_encoding"]
        inverted_board[inverted_board == -2] = player["WHITE"]["board_encoding"]
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
                if board[i, j] == player["EMPTY"]["board_encoding"]:
                    print("  O  ", end="")
                elif board[i, j] == player["BLACK"]["board_encoding"]:
                    print("  B  ", end="")
                else:
                    print("  W  ", end="")
                print("|", end="")
            print("")
            print(" " * (i * 3 + 1), end="")
            print("-" * (board.shape[1] * 7 - 1), end="")
            print("")

    def random_board(self,matrix):

        n = matrix.shape[0]
        m = np.random.randint(n//4, n-1)
        l = np.random.randint(n//4, n-1)

        # Randomly select the starting coordinates of the submatrix
        start_row = np.random.randint(0, n - m + 1)
        start_col = np.random.randint(0, n - l + 1)

        # Calculate the total number of elements in the submatrix
        total_elements = m * l
        total_nonzero = int((m * l * (0.5 + 0.5 * np.random.random()))//2) * 2 # //2 * 2 to always get even numbers => initialize board for black

        # Calculate the number of -1s and 1s to be placed in the submatrix
        num_minus_ones = total_nonzero // 2
        num_ones = total_nonzero - num_minus_ones
        num_zeros = total_elements - total_nonzero

        # Generate an array with equal numbers of -1s and 1s and shuffle it
        submatrix_values = np.array([player["BLACK"]["board_encoding"]] * num_minus_ones + 
                                    [player["WHITE"]["board_encoding"]] * num_ones + 
                                    [player["EMPTY"]["board_encoding"]] * num_zeros)
        
        np.random.shuffle(submatrix_values)

        # Reshape the shuffled values to match the submatrix dimensions
        shuffled_submatrix = submatrix_values.reshape((m, l))
        
        # Replace the submatrix with the shuffled values
        matrix[start_row:start_row + m, start_col:start_col + l] = shuffled_submatrix
        return matrix
