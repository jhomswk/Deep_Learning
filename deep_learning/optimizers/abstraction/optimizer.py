
class Optimizer(object):

    def __init__(self):
        self.params = set()
        
    def register(self, param):
        self.params.add(param)

    def unregister(self, param):
        self.params.discard(param)

    def update_params(self):
        for param in self.params:
            self.update(param)

    def update(self, param):
        raise NotImplementedError
