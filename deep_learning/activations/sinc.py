from .abstraction.activation import Activation
import numpy as np

class Sinc(Activation):

    def forward(self, x, train=False):
        if train: self.cache.append(x)
        return np.sinc(x)

    def backward(self, grad_output):
        x = self.cache.pop()
        denom = np.where(x != 0.0, x, 1.0)
        grad_input = (np.cos(np.pi*x) - np.sinc(x))/denom
        return grad_input*grad_output

