from .abstraction.activation import Activation
import numpy as np

class Selu(Activation):

    def __init__(self):
        super().__init__()
        self.alpha = 1.67326
        self.lam = 1.0507 

    def forward(self, x, train=False):
        output = np.array(x, dtype=float)
        self.update_output_with_negative_selu(x, output)
        output *= self.lam
        if train: self.cache.append(output)
        return output

    def update_output_with_negative_selu(self, x, output):
        negative = x < 0 
        output[negative] = self.alpha*(np.exp(x[negative]) - 1.0)

    def backward(self, grad_output):
        output = self.cache.pop()
        grad_input = np.where(output >= 0,
                self.lam, output + self.lam*self.alpha)
        return grad_input*grad_output

