from .abstraction.cost_function import Cost_Function
import numpy as np

class Reduce(Cost_Function):

    def __init__(self, cost_function, reducer, *args, **kwargs):
        self.cost_function = cost_function
        self.reduce = lambda x: reducer(x, *args, **kwargs)

    def eval(self, target, prediction):
        return self.reduce(self.cost_function.eval(target, prediction))

    def loss(self, target, prediction):
        return self.cost_function.loss(target, prediction)

    def grad(self, target, prediction):
        return self.cost_function.grad(target, prediction)

