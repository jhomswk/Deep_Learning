import numpy as np

class Random_Initializer:

    def __init__(self, seed=None):
        self.random_number_generator = (
                self.create_random_number_generator(seed))

    def __call__(self, shape, seed=None):
        rng = self.select_random_number_generator(seed)
        return self.query_random_number_generator(rng, shape)

    def select_random_number_generator(self, seed=None):
        return (self.random_number_generator if seed is None
                else self.create_random_number_generator(seed))

    def create_random_number_generator(self, seed):
        return np.random.RandomState(seed)

    def set_seed(self, seed):
        self.rng = self.create_random_number_generator(seed)

    def query_random_number_generator(self, rng, shape):
        raise NotImplementedError

    def set_variance(self, variance):
        raise NotImplementedError
