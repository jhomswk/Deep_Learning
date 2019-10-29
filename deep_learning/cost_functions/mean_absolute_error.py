from .abstraction.cost_function import Cost_Function
import numpy as np

class Mean_Absolute_Error(Cost_Function):

    def loss(self, target, prediction):
        return np.abs(target - prediction)

    def grad(self, target, prediction):
        return np.sign(prediction - target)
