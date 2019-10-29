from .abstraction.layer import Layer
import numpy as np


class Sequential_Layer(Layer):


    def __init__(self, block, name="Sequential"):
        super().__init__()
        self.output_sequence = True
        self.block = block
        self.name = name


    @property
    def input_size(self):
        return self.block.input_size


    def set_input_size(self, input_size):
        self.block.set_input_size(input_size)


    def set_optimizer(self, optimizer):
        self.block.set_optimizer(optimizer)


    def set_regularizer(self, regularizer):
        self.block.set_regularizer(regularizer)

    
    def init_recurrent_state(self, batch_size):
        return np.zeros((*np.atleast_1d(self.output_size), batch_size))


    def init_recurrent_gradient(self, batch_size):
        return np.zeros((*np.atleast_1d(self.output_size), batch_size))


    def forward(self, x, train):
        steps = x.shape[0]
        output = [None]*steps

        for t in range(steps):
            output[t] = self.block.forward(x[t], train)

        return np.array(output)


    def step_forward(self, x, state, train):
        output = self.block.forward(x, train)
        state = output
        return output, state


    def backward(self, grad_output):
        steps = grad_output.shape[0]
        grad_input = [None]*steps

        for t in reversed(range(steps)):
            grad_input[t] = self.block.backward(grad_output[t]) 

        return np.array(grad_input)


    def step_backward(self, grad_output, grad_state):
        grad_output = grad_output + grad_state
        grad_state = np.zeros_like(grad_state)
        grad_input = self.block.backward(grad_output)
        return grad_input, grad_state


    @property
    def is_trainable(self):
        return self.block.is_trainable


    @property
    def output_size(self):
        return self.block.output_size
        

    @property
    def params(self):
        yield from self.block.params


