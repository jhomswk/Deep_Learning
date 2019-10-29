from .abstraction.cost_function import Cost_Function
from ..util.sequenciate import sequenciate

class Sequential_Cost(Cost_Function):

    def __init__(self, cost_function):
        self.cost_function = cost_function

    def loss(self, target, prediction):
        return sequenciate(self.cost_function.loss, target, prediction)

    def grad(self, target, prediction):
        return sequenciate(self.cost_function.grad, target, prediction)
        
