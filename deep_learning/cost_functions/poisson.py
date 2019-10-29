from .abstraction.cost_function import Cost_Function
import numpy as np

class Poisson(Cost_Function):

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def loss(self, target, prediction):
        return prediction - target*np.log(prediction + self.epsilon)

    def grad(self, target, prediction):
        return 1.0 - target/(prediction + self.epsilon)
