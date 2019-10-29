from .abstraction.regularizer import Regularizer
from .ridge import Ridge
from .lasso import Lasso
import numpy as np

class Elastic(Regularizer):
    
    def __init__(self, lam1, lam2):
        super().__init__()
        self.lasso = Lasso(lam1)
        self.ridge = Ridge(lam2)
        self.lasso.params = self.params
        self.ridge.params = self.params

    def cost(self, param):
        return self.ridge.cost(param) + self.lasso.cost(param)

    def gradient(self, param):
        return self.ridge.gradient(param) + self.lasso.gradient(param)


