from .abstraction.cost_function import Cost_Function

class Null_Cost_Function(Cost_Function):

    def eval(self, target, prediction):
        return 0

    def loss(self, target, prediction):
        return 0

    def grad(self, target, prediction):
        return 0
