from ..util.exp_moving_average import Exp_Moving_Average
from .abstraction.optimizer import Optimizer
import numpy as np

class Adadelta(Optimizer):

    def __init__(self, lr=1.0, rho=0.95, eps=1e-8):
        super().__init__()
        self.delta_moment = dict()
        self.grad_moment = dict()
        self.rho = rho
        self.eps = eps
        self.lr = lr

    def register(self, param):
        super().register(param)
        self.delta_moment[param] = Exp_Moving_Average(self.rho) 
        self.grad_moment[param] = Exp_Moving_Average(self.rho)

    def unregister(self, param):
        super().unregister(param)
        self.delta_moment.pop(param, None)
        self.grad_moment.pop(param, None)

    def update(self, param):
        grad_moment = self.grad_moment.get(param)
        delta_moment = self.delta_moment.get(param)
        grad_moment.update(np.square(param.grad))
        delta = param.grad * np.sqrt(
                (delta_moment.average + self.eps)
              / (grad_moment.average + self.eps))
        delta_moment.update(np.square(delta))
        param.value = param.value - self.lr*delta




