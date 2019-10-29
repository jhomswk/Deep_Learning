from .abstraction.regularizer import Regularizer
import numpy as np

class Ridge(Regularizer):

    def __init__(self, lam):
        super().__init__()
        self.lam = lam 

    def cost(self, param):
        return 0.5*self.lam*np.sum(np.square(param.value))

    def gradient(self, param):
        return self.lam*param.value
