from ..regularizers.null_regularizer import Null_Regularizer
from ..optimizers.null_optimizer import Null_Optimizer
from ..initializers.random.normal import Normal
from ..util.parameter import Parameter
from .abstraction.layer import Layer
import numpy as np


class Embedding:


    def __init__(self, input_dim, output_dim, initializer=None,
                 reg_embedding=False, train=True, name="Embedding"):

        self.initializer = initializer or Normal(mean=0.0, stdev=1.0) 
        self.regularizer = Null_Regularizer()
        self.optimizer = Null_Optimizer()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.input_size = None

        self.embedding = Parameter()
        self.reset_embedding()

        self.reg_embedding = reg_embedding
        self._train = train

        self.name = name
        self.cache = []


    def reset_embedding(self):
        shape = (self.input_dim, self.output_dim)
        self.embedding.value = self.initializer(shape)
        self.embedding.grad = 0


    def load_embedding(self, embedding):
        self.input_dim, self.output_dim = embedding.shape
        self.embedding.value = embedding
        self.embedding.grad = 0


    def set_input_size(self, input_size):
        if self.input_size != input_size:
            self.input_size = input_size


    def set_optimizer(self, optimizer):
        if self.train:
            self.optimizer.unregister(self.embedding)
            optimizer.register(self.embedding)
        self.optimizer = optimizer


    def set_regularizer(self, regularizer):
        if self.reg_embedding:
            self.regularizer.unregister(self.embedding)
            regularizer.register(self.emmbedding)
        self.regularizer = regularizer


    def forward(self, x, train):
        output = np.moveaxis(self.embedding.value[x], -1, -2)
        if self.train and train: self.cache.append(x)
        return output


    def backward(self, grad_output):
        if self.train:
            x = self.cache.pop()
            grad_output = np.moveaxis(grad_output, -2, -1)
            self.embedding.grad = np.zeros_like(self.embedding.value)
            np.add.at(self.embedding.grad, x, grad_output)
        return None


    @property
    def train(self):
        return self._train


    @train.setter
    def train(self, train):
        if train != self._train:
            self._train = train
            if train: self.optimizer.register(self.embedding)
            else: self.optimizer.unregister(self.embedding)


    @property
    def is_trainable(self):
        return self._train


    @property
    def output_size(self):
        if self.input_size is None : return None
        shape = (*np.atleast_1d(self.input_size), self.output_dim)
        if len(shape) == 1: shape = shape[0]
        return shape


    @property
    def params(self):
        yield self.embedding


