from .abstraction.activation import Activation
import numpy as np

class Relu(Activation):

    def forward(self, x, train=False):
        scale = x >= 0
        if train: self.cache.append(scale)
        return scale*x

    def backward(self, grad_output):
        scale = self.cache.pop()
        return scale*grad_output

