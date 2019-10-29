import numpy as np

class Cost_Function:

    def eval(self, target, prediction):
        return np.mean(self.loss(target, prediction), axis=-1)

    def loss(self, target, prediction):
        raise NotImplementedError

    def grad(self, target, prediction):
        raise NotImplementedError
