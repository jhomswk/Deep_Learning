from .abstraction.cost_function import Cost_Function
import numpy as np

class Huber(Cost_Function):

    def __init__(self, delta=8):
        self.delta = delta

    def loss(self, target, prediction):
        dev = np.abs(target - prediction)
        quad = np.minimum(dev, self.delta)
        lin = dev - quad
        return np.square(quad)/2 + self.delta*lin

    def grad(self, target, prediction):
        return np.clip(prediction - target, -self.delta, self.delta)
