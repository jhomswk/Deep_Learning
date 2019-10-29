from .abstraction.regularizer import Regularizer

class Null_Regularizer(Regularizer):

    def __init__(self):
        self.params = None

    def register(self, param):
        pass

    def unregister(self, param):
        pass

    def cost(self, param):
        return 0

    def gradient(self, param):
        return 0

    def regularize_cost(self):
        return 0

    def regularize_params(self):
        pass
