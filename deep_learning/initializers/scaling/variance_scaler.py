from .variance_computer.input_variance_computer import Input_Variance_Computer
from .variance_computer.output_variance_computer import Output_Variance_Computer
from .variance_computer.average_variance_computer import Average_Variance_Computer
from ..random.normal import Normal
from ..random.uniform import Uniform


class Variance_Scaler:


    def __init__(self, dist, mode, scale, seed=None):
        self.computer = self.select_computer(mode, scale)
        self.initializer = self.select_initializer(dist, seed)
        self.mode = mode
        self.dist = dist


    def select_computer(self, mode, scale):
        if mode == "in":
            return Input_Variance_Computer(scale)

        if mode == "out":
            return Output_Variance_Computer(scale)

        if mode == "avg":
            return Average_Variance_Computer(scale)


    def select_initializer(self, dist, seed):
        if dist == "normal":
            return Normal(mean=0, stdev=None, seed=seed)

        if dist == "uniform":
            return Uniform(low=None, high=None, seed=seed)


    def __call__(self, shape, seed=None):
        variance = self.computer.compute_variance(shape)
        self.initializer.set_variance(variance)
        return self.initializer(shape, seed)


    def set_dist(self, dist):
        self.dist = dist
        self.intializer = self.select_initializer(dist, self.seed)


    def set_mode(self, mode):
        self.mode = mode
        self.computer = self.select_computer(mode, self.scale)


    def set_seed(self):
        self.initializer.set_seed(seed)


    def set_scale(self, scale):
        self.computer.set_scale(scale)


    @property
    def scale(self):
        return self.computer.scale


