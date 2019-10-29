from .abstraction.optimizer import Optimizer

class Null_Optimizer(Optimizer):

    def __init__(self):
        self.params = None

    def register(self, param):
        pass

    def unregister(self, param):
        pass

    def update_params(self):
        pass

    def update(self, param):
        pass
