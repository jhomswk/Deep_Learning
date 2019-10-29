from .abstraction.activation import Activation
import numpy as np

class Lrelu(Activation):

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, train=False):
        scale = np.where(x >= 0, 1.0, self.alpha)
        if train: self.cache.append(scale)
        return scale*x

    def backward(self, grad_output):
        scale = self.cache.pop()
        return scale*grad_output

