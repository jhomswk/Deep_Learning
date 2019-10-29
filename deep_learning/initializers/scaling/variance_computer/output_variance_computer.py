from .abstraction.variance_computer import Variance_Computer

class Output_Variance_Computer(Variance_Computer):

    def compute_variance(self, shape):
        return self.scale/self.fan_out(shape)
