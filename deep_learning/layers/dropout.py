from ..regularizers.null_regularizer import Null_Regularizer
from ..optimizers.null_optimizer import Null_Optimizer
from .abstraction.layer import Layer
from numpy.random import binomial

class Dropout(Layer):

    def __init__(self, prob, input_size=None, name="Dropout"):
        super().__init__()
        self.input_size = None
        self.prob = prob
        self.name = name
        self.cache = []


    def set_input_size(self, input_size):
        self.input_size = input_size


    def forward(self, x, train):
        if train:
            mask = binomial(1, 1-self.prob, size=x.shape[-2:])
            mask = mask/(1-self.prob)
            self.cache.append(mask)
            return mask*x
        return x


    def backward(self, grad_output):
        mask = self.cache.pop()
        return mask*grad_output


    @property
    def output_size(self):
        return self.input_size


    @property
    def is_trainable(self):
        return False

