from .abstraction.activation import Activation
import numpy as np

class Sqnonlin(Activation):

    def forward(self, x, train=False):
        clipped_x = np.clip(x, -2.0, 2.0)
        if train: self.cache.append(clipped_x)
        return clipped_x - np.sign(x)*np.square(clipped_x)/4.0

    def backward(self, grad_output):
        clipped_x = self.cache.pop()
        grad_input = 1.0 - np.abs(clipped_x)/2.0
        return grad_input*grad_output

