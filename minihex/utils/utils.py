#!/usr/bin/env python3
import copy

import torch
import torch.optim as optim

from collections import OrderedDict
from minihex.creation.create_model import create_model
from minihex.utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _one_pass(iters):
    for it in iters:
        try:
            yield next(it)
        except StopIteration:
            pass


def zip_list_of_lists(*iterables):
    iters = [iter(it) for it in iterables]
    output_list = []
    while True:
        iter_list = list(_one_pass(iters))
        output_list.extend(list(iter_list))
        if iter_list==[]:
            return output_list


def correct_position1d(position1d, board_size, player):
    if player:
        return position1d//board_size + (position1d%board_size)*board_size
    else:
        return position1d


def load_model(model_file, export_mode=False):
    checkpoint = torch.load(model_file, map_location=device)
    model = create_model(checkpoint['config'], export_mode)
    # breakpoint()

    # Error for some reason: dict keys have suffix 'internal_model' which is not recognised by the load_state_dict function.
    old_keys = checkpoint['model_state_dict'].keys()
    new_keys = [key[15:] for key in checkpoint['model_state_dict']]
    checkpoint['model_state_dict'] = OrderedDict((new_key, checkpoint['model_state_dict'][old_key]) for old_key, new_key in zip(old_keys, new_keys))
    # checkpoint['model_state_dict'].update(renamed_ordered_dict)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    torch.no_grad()
    return model


def create_optimizer(optimizer_type, parameters, learning_rate, momentum, weight_decay):
    logger.debug("=== creating optimizer ===")
    if optimizer_type == 'adadelta':
        return optim.Adadelta(parameters, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        return optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        logger.error(f'Unknown optimizer {optimizer_type}')
        raise SystemExit


def get_targets(boards, gamma):
    target_list = [[0.5 + 0.5 * (-1) ** k * (1 - gamma) ** (2 * (k//2)) for k in reversed(range(len(
        board.move_history)))] for board in boards]
    return torch.tensor(zip_list_of_lists(*target_list), device=torch.device('cpu'))


def merge_dicts_of_dicts(dict1, dict2):
    output_dict = copy.deepcopy(dict1)
    for key, sub_dict in dict2.items():
        output_dict[key].update(sub_dict)
    return output_dict


class Average:
    def __init__(self):
        self.num_samples = 0
        self.total = 0.0

    def add(self, value, num_samples):
        self.num_samples += num_samples
        self.total += value

    def mean(self):
        try:
            return self.total / self.num_samples
        except ZeroDivisionError:
            return float("NaN")
