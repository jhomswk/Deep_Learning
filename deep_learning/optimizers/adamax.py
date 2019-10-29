from ..util.exp_moving_average import Exp_Moving_Average
from ..util.moving_max_norm import Moving_Max_Norm
from .abstraction.optimizer import Optimizer
import numpy as np

class Adamax(Optimizer):

    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.momentum = dict()
        self.max_norm = dict()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr

    def register(self, param):
        super().register(param)
        self.momentum[param] = Exp_Moving_Average(self.beta1)
        self.max_norm[param] = Moving_Max_Norm(self.beta2)

    def unregister(self, param):
        super().unregister(param)
        self.momentum.pop(param, None)
        self.max_norm.pop(param, None)

    def update(self, param):
        momentum = self.momentum.get(param)
        momentum.update(param.grad)
        momentum = momentum.get_corrected_average()

        max_norm = self.max_norm.get(param)
        max_norm.update(param.grad)
        max_norm = max_norm.norm

        gain = self.lr/(max_norm + self.eps)
        param.value = param.value - gain*momentum

