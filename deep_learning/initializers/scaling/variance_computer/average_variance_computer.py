from .abstraction.variance_computer import Variance_Computer

class Average_Variance_Computer(Variance_Computer):

    def compute_variance(self, shape):
        return self.scale/self.average_fan(shape)

    def average_fan(self, shape):
        return (self.fan_in(shape) + self.fan_out(shape))/2
