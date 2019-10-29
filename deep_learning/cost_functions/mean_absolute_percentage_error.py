from .abstraction.cost_function import Cost_Function
import numpy as np

class Mean_Absolute_Percentage_Error(Cost_Function):

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def loss(self, target, prediction):
        dev = np.abs(target - prediction)
        return 100*dev/self.clip(np.abs(target))

    def grad(self, target, prediction):
        grad = np.sign(prediction - target)
        return 100*grad/self.clip(np.abs(target))

    def clip(self, x):
        return np.clip(x, self.epsilon, None)
