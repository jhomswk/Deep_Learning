import numpy as np

class Variance_Computer:

    def __init__(self, scale):
        self.scale = scale

    def compute_variance(self, shape):
        raise NotImplementedError

    def set_scale(self, scale):
        self.scale = scale

    def fan_in(self, shape):
        return np.prod(shape[1:])

    def fan_out(self, shape):
        return np.prod(shape[:-1])
