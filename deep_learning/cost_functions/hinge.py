from .abstraction.cost_function import Cost_Function
import numpy as np

class Hinge(Cost_Function):

    def __init__(self, delta=1.0):
        self.delta = delta

    def loss(self, target, prediction):
        return np.maximum(0.0, self.delta - target*prediction)

    def grad(self, target, prediction):
        return np.where(self.delta - target*prediction >= 0, -target, 0)

