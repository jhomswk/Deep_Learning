from ..util.exp_moving_average import Exp_Moving_Average
from .abstraction.optimizer import Optimizer
import numpy as np

class RMSProp(Optimizer):

    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__()
        self.momentum = dict()
        self.beta = beta
        self.eps = eps
        self.lr = lr

    def register(self, param):
        super().register(param)
        self.momentum[param] = Exp_Moving_Average(self.beta)

    def unregister(self, param):
        super().unregister(param)
        self.momentum.pop(param, None)

    def update(self, param):
        momentum = self.momentum.get(param)
        momentum.update(np.square(param.grad))
        momentum = momentum.average
        gain = self.lr/(np.sqrt(momentum) + self.eps)
        param.value = param.value - gain*param.grad

