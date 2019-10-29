import numpy as np

class Regularizer:

    def __init__(self):
        self.params = set()


    def register(self, param):
        self.params.add(param)


    def unregister(self, param):
        self.params.discard(param)


    def cost(self, param):
        raise NotImplementedError


    def gradient(self, param):
        raise NotImplementedError


    def normalized_cost(self, param):
        param_cost = self.cost(param)
        norm = np.prod(param.value.shape)
        return param_cost/norm


    def regularize_cost(self):
        reg_cost = sum(map(self.normalized_cost, self.params))
        norm = len(self.params) if self.params else 1
        return reg_cost/norm


    def regularize_params(self):
        for param in self.params:
            param.grad += self.gradient(param)



