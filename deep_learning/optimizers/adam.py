from ..util.exp_moving_average import Exp_Moving_Average
from .abstraction.optimizer import Optimizer
import numpy as np

class Adam(Optimizer):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.mom1 = dict()
        self.mom2 = dict()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr

    def register(self, param):
        super().register(param)
        self.mom1[param] = Exp_Moving_Average(self.beta1)
        self.mom2[param] = Exp_Moving_Average(self.beta2)

    def unregister(self, param):
        super().unregister(param)
        self.mom1.pop(param, None)
        self.mom2.pop(param, None)

    def update(self, param):
        mom1 = self.mom1.get(param)
        mom2 = self.mom2.get(param)
        mom1.update(param.grad)
        mom2.update(np.square(param.grad))
        mom1 = mom1.get_corrected_average()
        mom2 = mom2.get_corrected_average()
        gain = self.lr/(np.sqrt(mom2) + self.eps)
        param.value = param.value - gain*mom1
        
