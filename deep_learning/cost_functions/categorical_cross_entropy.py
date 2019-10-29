from .abstraction.cost_function import Cost_Function
import numpy as np

class Categorical_Cross_Entropy(Cost_Function):
    
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def loss(self, target, prediction):
        prediction = self.clip(prediction)
        loss = -target*np.log(prediction)
        return np.sum(loss, axis=0, keepdims=True)

    def grad(self, target, prediction):
        prediction = self.clip(prediction)
        return -target/prediction 

    def clip(self, x):
        return np.clip(x, self.epsilon, 1 - self.epsilon)
