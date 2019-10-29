from .abstraction.score import Score
import numpy as np


class Dot_Product(Score):


    def __init__(self):
        self.scale = None
        self.cache = []


    def set_input_size(self, input_size, state_size):
        assert input_size == state_size
        size = np.atleast_1d(input_size)
        self.scale = 1.0/np.sqrt(size[-1])


    def forward(self, x, state, train):
        state = state[0] if isinstance(state, tuple) else state
        scores = self.scale*np.sum(state*x, axis=-2, keepdims=True)
        if train: self.cache.append((x, state))
        return scores


    def backward(self, grad_scores):
        input, state = self.cache.pop()
        grad_state = self.scale*np.sum(grad_scores*input, axis=0)
        grad_input = self.scale*grad_scores*state
        return grad_input, grad_state

