from .abstraction.cost_function import Cost_Function
import numpy as np

class Binary_Cross_Entropy(Cost_Function):

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def loss(self, target, prediction):
        prediction = self.clip(prediction)
        return -(target*np.log(prediction)
                + (1.0 - target)*np.log(1.0-prediction))

    def grad(self, target, prediction):
        prediction = self.clip(prediction)
        return -target/prediction + (1.0 - target)/(1.0 - prediction)

    def clip(self, x):
        return np.clip(x, self.epsilon, 1.0 - self.epsilon)
