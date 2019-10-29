from .abstraction.activation import Activation
import numpy as np

class Arctan(Activation):

    def forward(self, x, train=False):
        if train: self.cache.append(x)
        return np.arctan(x)

    def backward(self, grad_output):
        x = self.cache.pop()
        return grad_output/(1.0 + np.square(x))

