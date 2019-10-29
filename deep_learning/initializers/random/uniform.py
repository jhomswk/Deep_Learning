from ..abstraction.random_initializer import Random_Initializer
import numpy as np

class Uniform(Random_Initializer):

    def __init__(self, low, high, seed=None):
        super().__init__(seed)
        self.low = low 
        self.high = high

    def query_random_number_generator(self, rng, shape):
        return rng.uniform(low=self.low, high=self.high, size=shape)

    def set_variance(self, variance):
        lim = np.sqrt(3*variance)
        self.low = -lim
        self.high = lim

