from .abstraction.optimizer import Optimizer

class Gradient_Descent(Optimizer):

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update(self, param):
        param.value = param.value - self.lr*param.grad

