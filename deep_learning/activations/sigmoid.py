from .abstraction.activation import Activation
import numpy as np

class Sigmoid(Activation):

    def forward(self, x, train=False):
        output = np.zeros_like(x, dtype=float)
        self.update_output_with_positive_sigmoid(x, output)
        self.update_output_with_negative_sigmoid(x, output)
        if train: self.cache.append(output)
        return output

    def update_output_with_positive_sigmoid(self, x, output):
        positive = x >= 0
        output[positive] = 1.0/(1.0 + np.exp(-x[positive]))
        
    def update_output_with_negative_sigmoid(self, x, output):
        negative = x < 0
        exp_x = np.exp(x[negative])
        output[negative] = exp_x/(1 + exp_x)

    def backward(self, grad_output):
        output = self.cache.pop()
        return output*(1 - output)*grad_output
