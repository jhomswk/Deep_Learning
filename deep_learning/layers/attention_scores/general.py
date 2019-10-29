from ..sequential_layer import Sequential_Layer as Sequential
from .abstraction.score import Score
from ..dense import Dense
import numpy as np


class General(Score):


    def __init__(self, units=None, block=None):

        assert block or units

        block = block or Dense(units)
        self.block = Sequential(block)
        self.cache = []


    def set_input_size(self, input_size, state_size):
        if self.input_size != input_size:
            self.block.set_input_size(input_size)

        assert state_size == self.block.output_size


    def set_optimizer(self, optimizer):
        self.block.set_optimizer(optimizer)


    def set_regularizer(self, regularizer):
        self.block.set_regularizer(regularizer)


    def forward(self, x, state, train):
        state = state[0] if isinstance(state, tuple) else state
        weights = self.block.forward(x, train)
        scores = np.sum(weights*state, axis=-2, keepdims=True)
        if train: self.cache.append((weights, state))
        return scores


    def backward(self, grad_scores):
        weights, state = self.cache.pop()
        grad_weights = grad_scores*state
        grad_state = np.mean(grad_scores*weights, axis=0)
        grad_input = self.block.backward(grad_weights)
        return grad_input, grad_state

    
    @property
    def input_size(self):
        return self.block.input_size


    @property
    def state_size(self):
        return self.block.output_size


    @property
    def params(self):
        for block in self.blocks:
            yield from block.params


