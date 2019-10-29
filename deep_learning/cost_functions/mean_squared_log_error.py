from .abstraction.cost_function import Cost_Function
import numpy as np

class Mean_Squared_Log_Error(Cost_Function):

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def loss(self, target, prediction):
        target = self.clip(target)
        prediction = self.clip(prediction)
        return np.square(np.log1p(target) - np.log1p(prediction))

    def grad(self, target, prediction):
        target = self.clip(target)
        prediction = self.clip(prediction)
        grad = 2.0*(np.log1p(prediction) - np.log1p(target))
        return grad/(1.0 + prediction)

    def clip(self, x):
        return np.clip(x, self.epsilon, None)
