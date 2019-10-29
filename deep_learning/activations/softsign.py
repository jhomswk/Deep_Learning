from .abstraction.activation import Activation
import numpy as np

class Softsign(Activation):

    def forward(self, x, train=False):
        if train: self.cache.append(x)
        return x/(1.0 + np.abs(x))

    def backward(self, grad_output):
        x = self.cache.pop()
        grad_input = np.square(1.0/(1.0 + np.abs(x)))
        return grad_input*grad_output

