#!/usr/bin/env python3
from configparser import ConfigParser
import numpy as np
from minihex.interactive.gui import Gui
# from hexhex.logic.hexboard import Board
# from hexhex.logic.hexgame import MultiHexGame
# from hexhex.utils.utils import load_model

# from hexhex.model.hexconvolution import RandomModel

class InteractiveGame:
    """
    allows to play a game against a model
    """

    def __init__(self, board):

        config = ConfigParser()
        config.read('config.ini')
        
        self.config = config
        self.switch_allowed = False # self.config.getboolean("INTERACTIVE", 'switch', fallback=True)

        self.board = board
        self.board_size = self.board.shape[0]
        self.gui = Gui(self.board, self.config.getint("INTERACTIVE", 'gui_radius', fallback=50),
                       self.config.getboolean("INTERACTIVE", 'dark_mode', fallback=False))
        
    def play_move(self):
        move = self.get_move()
        print(move)
        if move == 'show_ratings':
            self.gui.show_field_text = not self.gui.show_field_text
        elif move == 'redraw':
            pass
        elif move == 'ai_move':
            self.play_ai_move()
        elif move == 'undo_move':
            self.undo_move()
        return move
    
    def undo_move(self):
        self.board.undo_move_board()
        # forgot know what the ratings were better not show anything
        self.gui.last_field_text = None
        self.gui.update_board(self.board)

    def play_ai_move(self):
        move_ratings = self.game.batched_single_move(self.model)
        rating_strings = []
        for rating in move_ratings[0]:
            if rating > 99:
                rating_strings.append('+')
            elif rating < -99:
                rating_strings.append('-')
            else:
                rating_strings.append("{0:0.1f}".format(rating))
        self.gui.update_field_text(rating_strings)

        if self.board.winner:
            self.gui.set_winner("agent has won!")

    def get_move(self):
        # actions = np.arange(self.board.shape[0] * self.board.shape[1])
        valid_actions = np.where(self.board == 0)
        print(self.board)
        valid_actions = list(zip(valid_actions[0], valid_actions[1]))
        # print(valid_actions)
        while True:
            move = self.gui.get_move()
            if move in ['ai_move', 'undo_move', 'redraw', 'show_ratings', 'restart']:
                return move
            move = tuple(map(sum, zip(move, (-1,-1))))
            if move in valid_actions:
                return move
    
    def choose_action(self, board, action_mask = None):
        self.board = board
        action = self.play_move()
        action = self.coordinate_to_action(action)

        # self.winner = self.simulator.fast_move(action)
        return action

    def coordinate_to_action(self, coords):
        return np.ravel_multi_index(coords, (self.board_size, self.board_size))

def play_game(interactive):
    while True:
        interactive.play_move()
        if interactive.board.winner:
            break


def main():
    config = ConfigParser()
    config.read('config.ini')

    while True:
        interactive = InteractiveGame(config)
        play_game(interactive)
        interactive.gui.wait_for_pressing_r()  # wait for 'r' to start new game


if __name__ == '__main__':
    main()
