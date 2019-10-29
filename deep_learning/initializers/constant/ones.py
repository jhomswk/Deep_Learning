from ..abstraction.initializer import Initializer
import numpy as np

class Ones(Initializer):

    def __call__(self, shape):
        return np.ones(shape)
