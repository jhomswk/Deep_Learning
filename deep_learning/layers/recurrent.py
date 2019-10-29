from .abstraction.layer import Layer
from ..activations.tanh import Tanh
from .dense import Dense
import numpy as np


class Recurrent(Layer):


    def __init__(self, units=None, input_block=None,
                 state_block=None, activ_block=None,
                 output_sequence=True, name="Recurrent"):

        assert input_block and state_block or units

        self.input_block = input_block or Dense(units)
        self.state_block = state_block or Dense(units)
        self.activ_block = activ_block or Tanh()
        self.output_sequence = output_sequence
        self.name = name
        self.steps = 0


    @property
    def blocks(self):
        yield self.input_block
        yield self.state_block
        yield self.activ_block


    def set_input_size(self, input_size):
        if input_size != self.input_size:
            self.input_block.set_input_size(input_size)
            self.activ_block.set_input_size(self.input_block.output_size)
            self.state_block.set_input_size(self.activ_block.output_size)
            assert (self.state_block.output_size == self.input_block.output_size)


    def set_optimizer(self, optimizer):
        for block in self.blocks:
            block.set_optimizer(optimizer)
    

    def set_regularizer(self, regularizer):
        for block in self.blocks:
            block.set_regularizer(regularizer)


    def init_recurrent_state(self, batch_size):
        return np.zeros((*np.atleast_1d(self.output_size), batch_size))


    def init_recurrent_gradient(self, batch_size):
        return np.zeros((*np.atleast_1d(self.output_size), batch_size))


    def forward(self, x, train):
        steps = x.shape[0]
        outputs = [None]*steps
        batch_size = x.shape[-1]
        state = self.init_recurrent_state(batch_size)

        for t in range(steps):
            output, state = self.step_forward(x[t], state, train)
            outputs[t] = output

        return np.array(outputs) if self.output_sequence else output 


    def step_forward(self, x, state, train):
        state = self.state_block.forward(state, train)
        state = state + self.input_block.forward(x, train)
        state = self.activ_block.forward(state, train)
        output = state
        if train: self.steps += 1
        return output, state


    def backward(self, grad_output):
        steps = self.steps 
        grad_input = [None]*steps
        batch_size = grad_output.shape[-1]
        grad_state = self.init_recurrent_gradient(batch_size)

        if not self.output_sequence:
            grad_out = [0]*steps
            grad_out[-1] = grad_output
            grad_output = grad_out

        for t in reversed(range(steps)):
            grad_input[t], grad_state = (
                    self.step_backward(grad_output[t], grad_state))
            
        grad_input = np.array(grad_input)
        return grad_input


    def step_backward(self, grad_output, grad_state):
        grad_out = grad_output + grad_state
        grad_out = self.activ_block.backward(grad_out)
        grad_input = self.input_block.backward(grad_out)
        grad_state = self.state_block.backward(grad_out)
        self.steps -= 1
        return grad_input, grad_state

    
    @property
    def is_trainable(self):
        return True


    @property
    def input_size(self):
        return self.input_block.input_size


    @property
    def output_size(self):
        return self.activ_block.output_size


    @property
    def params(self):
        for block in self.blocks:
            yield from block.params
