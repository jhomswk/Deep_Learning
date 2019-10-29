from ..regularizers.null_regularizer import Null_Regularizer
from ..optimizers.null_optimizer import Null_Optimizer
from .abstraction.layer import Layer
import numpy as np


class Block(object):


    def __init__(self, layers=None, name="Block"):
        super().__init__()
        self.regularizer = Null_Regularizer()
        self.optimizer = Null_Optimizer()
        self.input_size = None
        self.layers = []
        self.name = name

        layers = layers or []
        for layer in layers:
            self.add(layer)


    def add(self, layer):
        self.setup_layer(layer)
        self.layers.append(layer)


    def setup_layer(self, layer):
        if self.output_size:
            layer.set_input_size(self.output_size)
        layer.set_optimizer(self.optimizer)
        layer.set_regularizer(self.regularizer)


    def set_input_size(self, input_size):
        if self.input_size != input_size:
            self.input_size = input_size
            for layer in self.layers:
                layer.set_input_size(input_size)
                input_size = layer.output_size


    @property
    def output_size(self):
        return (self.input_size if not self.layers
                else self.layers[-1].output_size)
        

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        for layer in self.layers:
            layer.set_optimizer(optimizer)


    def set_regularizer(self, regularizer):
        self.regularizer = regularizer
        for layer in self.layers:
            layer.set_regularizer(regularizer)


    def init_recurrent_state(self, batch_size):
        return [layer.init_recurrent_state(batch_size)
                for layer in self.layers]


    def init_recurrent_gradient(self, batch_size):
        return [layer.init_recurrent_gradient(batch_size)
                for layer in self.layers]

    
    def forward(self, x, train):
        output = x
        for layer in self.layers:
            output = layer.forward(output, train)
        return output


    def step_forward(self, x, state, train):
        output, state = x, state[:]
        for l, layer in enumerate(self.layers):
            output, state[l] = (
                    layer.step_forward(output, state[l], train))
        return output, state


    def backward(self, grad_output):
        grad_input = grad_output
        for layer in reversed(self.layers):
            grad_input = layer.backward(grad_input)
        return grad_input


    def step_backward(self, grad_output, grad_state, train):
        grad_input, grad_state = grad_output, grad_state[:]
        for l, layer in enumerate(self.layers):
            grad_input, grad_state[l] = (
                    layer.step_backward(grad_input, grad_state[l]))
        return grad_input, grad_state 


    @property
    def is_trainable(self):
        return any(map(lambda x: x.is_trainable, self.layers))


    @property
    def params(self):
        for layer in self.layers: yield from layer.params

