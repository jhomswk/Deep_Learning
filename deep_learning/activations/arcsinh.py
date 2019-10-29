from .abstraction.activation import Activation
import numpy as np

class Arcsinh(Activation):

    def forward(self, x, train=False):
        if train: self.cache.append(x)
        return np.arcsinh(x)

    def backward(self, grad_output):
        x = self.cache.pop()
        return grad_output/np.hypot(1, x)

