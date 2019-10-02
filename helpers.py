import random
from numbers import Number

import numpy as np
import torch


def set_all_rng_states(state: dict):
    """Set RNG states
    Sets the states of RNGs that I know of and may use, in case we wish to resume the exact state
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['pytorch'])
    if 'pytorch_cuda' in state:
        torch.cuda.set_rng_state_all(state['pytorch_cuda'])


def set_all_rng_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    # see PyTorch Notes
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


class Meter:
    """Meter to track mean"""
    def __init__(self):
        self.__count = 0.
        self.__total = 0.

    def update(self, value: Number, count: int):
        self.__total += value
        self.__count += count

    @property
    def mean(self) -> Number:
        return self.__total / self.__count

    @property
    def total(self) -> Number:
        return self.__total
