from .abstraction.activation import Activation
import numpy as np

class Softplus(Activation):

    def __init__(self, limit=30):
        super().__init__()
        self.limit = limit

    def forward(self, x, train=False):
        exp = np.zeros_like(x, dtype=float)
        self.update_exponential_over_window(x, exp)
        if train: self.cache.append((x, exp))
        return np.log(1.0 + exp) + np.maximum(x, 0.0)

    def update_exponential_over_window(self, x, exp):
        window = self.extract_window(x)
        exp[window] = np.exp(-np.abs(x[window]))

    def extract_window(self, x):
        return np.logical_and(-self.limit < x, x < self.limit)

    def backward(self, grad_output):
        x, exp = self.cache.pop()
        grad_input = np.heaviside(x, 0.0)
        grad_input -= np.sign(x)*exp/(1.0 + exp)
        return grad_input*grad_output

