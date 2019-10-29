from .abstraction.activation import Activation
import numpy as np

class Bid(Activation):

    def forward(self, x, trian=False):
        if train: self.cache.append(x)
        return x + (np.hypot(1.0, x) - 1.0)/2.0

    def backward(self, grad_output):
        x = self.cache.pop()
        grad_input = 1.0 + x/(2.0*np.hypot(1.0, x))
        return grad_input*grad_output

