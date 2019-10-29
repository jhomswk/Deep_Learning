from ..regularizers.null_regularizer import Null_Regularizer
from ..optimizers.null_optimizer import Null_Optimizer
from .abstraction.layer import Layer
import numpy as np

class Flatten(Layer):

    def __init__(self, name="Flatten"):
        super().__init__()
        self.input_size = None
        self.name = name


    def set_input_size(self, input_size):
        self.input_size = input_size


    def forward(self, x, train=False):
        return x.reshape(self.output_size, -1)


    def backward(self, grad_output):
        return grad_output.reshape(*self.input_size, -1)


    @property
    def output_size(self):
        return self.input_size and np.prod(self.input_size)


    @property
    def is_trainable(self):
        return False

