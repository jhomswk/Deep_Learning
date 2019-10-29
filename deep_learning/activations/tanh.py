from .abstraction.activation import Activation
import numpy as np

class Tanh(Activation):

    def forward(self, x, train=False):
        output = np.tanh(x)
        if train: self.cache.append(output)
        return output

    def backward(self, grad_output):
        output = self.cache.pop()
        return (1 - np.square(output))*grad_output

