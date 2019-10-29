from ..regularizers.null_regularizer import Null_Regularizer
from ..optimizers.null_optimizer import Null_Optimizer
from ..initializers.scaling.glorot import Glorot
from ..initializers.constant.zeros import Zeros 
from ..util.moving_average import Moving_Average
from ..util.parameter import Parameter
from .abstraction.layer import Layer
import numpy as np

class Dense(Layer):


    def __init__(self, units, weight_init = None, bias_init = None,
            reg_weight = True, reg_bias = False, name = "Dense"):

        self.regularizer = Null_Regularizer() 
        self.optimizer = Null_Optimizer() 

        self.weight_init = weight_init or Glorot("normal")
        self.weight_grad = Moving_Average()
        self.weight = Parameter()
        self.reg_weight = reg_weight

        self.bias_init = bias_init or Zeros()
        self.bias_grad = Moving_Average()
        self.bias = Parameter() 
        self.reg_bias = reg_bias

        self.input_size = None
        self.units = units 
        self.name = name
        self.cache = []



    def set_input_size(self, input_size):
        if self.input_size != input_size:
            self.input_size = input_size
            self.setup_parameters()


    def setup_parameters(self):
        self.weight.value = self.weight_init(self.weight_shape)
        self.bias.value = self.bias_init(self.bias_shape)


    @property
    def weight_shape(self):
        return (self.output_size, self.input_size) 


    @property
    def bias_shape(self):
        return (self.output_size, 1)

    
    def set_optimizer(self, optimizer):
        self.optimizer.unregister(self.weight)
        self.optimizer.unregister(self.bias)
        optimizer.register(self.weight)
        optimizer.register(self.bias)
        self.optimizer = optimizer


    def set_regularizer(self, regularizer):
        if self.reg_weight:
            self.regularizer.unregister(self.weight)
            regularizer.register(self.weight)
        if self.reg_bias:
            self.regularizer.unregister(self.bias)
            regularizer.register(self.bias)
        self.regularizer = regularizer


    def forward(self, x, train):
        if train: self.cache.append(x)
        return np.dot(self.weight.value, x) + self.bias.value


    def backward(self, grad_output):
        self.weight.grad = self.compute_grad_weight(grad_output)
        self.bias.grad = self.compute_grad_bias(grad_output)
        grad_input = self.compute_grad_input(grad_output)

        self.weight_grad.update(self.weight.grad)
        self.bias_grad.update(self.bias.grad)

        del self.cache[-1]

        if len(self.cache) == 0:
            self.weight.grad = self.weight_grad.average
            self.bias.grad = self.bias_grad.average
            self.weight_grad.reset()
            self.bias_grad.reset()

        return grad_input


    def compute_grad_weight(self, grad_output):
        input, batch_size = self.cache[-1], grad_output.shape[-1]
        return np.dot(grad_output, input.T)/batch_size


    def compute_grad_bias(self, grad_output):
        return np.mean(grad_output, axis=1, keepdims=True)


    def compute_grad_input(self, grad_output):
        return np.dot(self.weight.value.T, grad_output)


    @property
    def params(self):
        yield self.weight
        yield self.bias


    @property
    def output_size(self):
        return self.units


    @property
    def is_trainable(self):
        return True

