from .abstraction.cost_function import Cost_Function
import numpy as np

class Mean_Squared_Error(Cost_Function):

    def loss(self, target, prediction):
        return np.square(target - prediction)

    def grad(self, target, prediction):
        return 2.0*(prediction - target)
