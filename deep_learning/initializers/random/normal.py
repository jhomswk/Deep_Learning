from ..abstraction.random_initializer import Random_Initializer
import numpy as np

class Normal(Random_Initializer):

    def __init__(self, mean, stdev, seed=None):
        super().__init__(seed)
        self.mean = mean
        self.stdev = stdev

    def query_random_number_generator(self, rng, shape):
        return rng.normal(loc=self.mean, scale=self.stdev, size=shape)

    def set_variance(self, variance):
        self.stdev = np.sqrt(variance)
