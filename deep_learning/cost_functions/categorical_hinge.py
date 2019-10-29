from .abstraction.cost_function import Cost_Function
import numpy as np

class Categorical_Hinge(Cost_Function):

    def __init__(self, delta=1.0):
        self.delta = delta

    def loss(self, target, prediction):
        loss = np.maximum(0.0, self.margin(target, prediction))
        return np.sum(loss, axis=0, keepdims=True)

    def grad(self, target, prediction):
        grad = np.heaviside(self.margin(target, prediction), 0)
        return np.where(target == 1, -np.sum(grad, axis=0), grad)

    def margin(self, target, prediction):
        correct = np.max(target*prediction, axis=0)
        margin = prediction - correct + self.delta
        margin[target==1] = 0
        return margin

