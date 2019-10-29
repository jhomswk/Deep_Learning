from .abstraction.activation import Activation
import numpy as np

class Softmax(Activation):

    def forward(self, x, train=False):
        exp_x = np.exp(x - np.amax(x, axis=0))
        output = exp_x/np.sum(exp_x, axis=0)
        if train: self.cache.append(output)
        return output

    def backward(self, grad_output):
        output = self.cache.pop()
        dot = np.sum(grad_output*output, axis=0, keepdims=True)
        return output*(grad_output - dot)

