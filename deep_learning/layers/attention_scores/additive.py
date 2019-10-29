from ..sequential_layer import Sequential_Layer as Sequential
from .abstraction.score import Score
from ...activations.tanh import Tanh
from ..dense import Dense
from ..block import Block
import numpy as np


class Additive(Score):


    def __init__(self, units=None, input_block=None,
                 state_block=None, score_block=None):

        assert input_block and state_block or units

        self.input_block = input_block or Dense(units)
        self.state_block = state_block or Dense(units)
        self.score_block = score_block or Block([Tanh(), Dense(1)])

        self.input_block = Sequential(self.input_block)
        self.score_block = Sequential(self.score_block)

        self.units = None


    @property
    def blocks(self):
        yield self.input_block
        yield self.state_block
        yield self.score_block


    def set_input_size(self, input_size, state_size):
        if self.input_size != input_size:
            self.input_block.set_input_size(input_size)

        if self.state_size != state_size:
            self.state_block.set_input_size(state_size)

        units = self.state_block.output_size

        if self.units != units:
            self.units = units
            self.score_block.set_input_size(units)

        assert (self.input_block.output_size
                == self.state_block.output_size)


    def set_optimizer(self, optimizer):
        for block in self.blocks:
            block.set_optimizer(optimizer)


    def set_regularizer(self, regularizer):
        for block in self.blocks:
            block.set_regularizer(regularizer)


    def forward(self, x, state, train):
        state = state[0] if isinstance(state, tuple) else state
        score_x = self.input_block.forward(x, train)
        score_h = self.state_block.forward(state, train)
        scores = self.score_block.forward(score_x + score_h, train)
        return scores


    def backward(self, grad_scores):
        grad_sum = self.score_block.backward(grad_scores)
        grad_score_h = np.mean(grad_sum, axis=0)
        grad_score_x = grad_sum
        grad_state = self.state_block.backward(grad_score_h)
        grad_input = self.input_block.backward(grad_score_x)
        return grad_input, grad_state


    @property
    def input_size(self):
        return self.input_block.input_size


    @property
    def state_size(self):
        return self.state_block.input_size


    @property
    def output_size(self):
        return self.score_block.output_size


    @property
    def params(self):
        for block in self.blocks:
            yield from block.params


