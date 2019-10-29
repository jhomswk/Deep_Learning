from .abstraction.regularizer import Regularizer
import numpy as np

class Lasso(Regularizer):

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def cost(self, param):
        return self.lam*np.sum(np.abs(param.value))

    def gradient(self, param):
        return self.lam*np.sign(param.value)
