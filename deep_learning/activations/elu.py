from .abstraction.activation import Activation
import numpy as np

class Elu(Activation):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, train=False):
        output = np.array(x, dtype=float)
        self.update_output_with_negative_elu(x, output)
        if train: self.cache.append(output)
        return output

    def update_output_with_negative_elu(self, x, output):
        negative = x < 0 
        output[negative] = self.alpha*(np.exp(x[negative]) - 1.0)

    def backward(self, grad_output):
        output = self.cache.pop()
        grad_input = np.where(output >= 0, 1.0, output + self.alpha)
        return grad_input*grad_output

