from ..util.exp_moving_average import Exp_Moving_Average
from .abstraction.optimizer import Optimizer

class Momentum(Optimizer):
    
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__()
        self.momentum = dict()
        self.beta = beta
        self.lr = lr

    def register(self, param):
        super().register(param)
        self.momentum[param] = Exp_Moving_Average(self.beta)

    def unregister(self, param):
        super().unregister(param)
        self.momentum.pop(param, None)

    def update(self, param):
        momentum = self.momentum.get(param)
        momentum.update(param.grad)
        momentum = momentum.average
        param.value = param.value - self.lr*momentum

