from .abstraction.activation import Activation
import numpy as np

class Gaussian(Activation):

    def forward(self, x, train=False):
        output = np.exp(-np.square(x))
        if train: self.cache.append(-2*x*output)
        return output

    def backward(self, grad_output):
        grad_input = self.cache.pop()
        return grad_input*grad_output

