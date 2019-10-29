from ..abstraction.initializer import Initializer
import numpy as np

class Constant(Initializer):

    def __init__(self, constant):
        self.constant = constant

    def __call__(self, shape):
        return np.full(shape, self.constant)

    def set_constant(self, constant):
        self.constant = constant
