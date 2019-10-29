from .abstraction.activation import Activation
import numpy as np

class Sin(Activation):

    def forward(self, x, train=False):
        if train: self.cache.append(x)
        return np.sin(x)

    def backward(self, grad_output):
        x = self.cache.pop()
        return np.cos(x)*grad_output

