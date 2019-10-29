from .abstraction.cost_function import Cost_Function
import numpy as np

class Logcosh(Cost_Function):

    def loss(self, target, prediction):
        dev = np.abs(target - prediction)
        return dev + np.log(1.0 + np.exp(-2.0*dev)) - np.log(2.0)

    def grad(self, target, prediction):
        return np.tanh(prediction - target)
