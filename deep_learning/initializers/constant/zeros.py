from ..abstraction.initializer import Initializer
import numpy as np

class Zeros(Initializer):

    def __call__(self, shape):
        return np.zeros(shape)
