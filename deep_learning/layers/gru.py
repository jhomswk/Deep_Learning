from ..activations.sigmoid import Sigmoid
from ..activations.tanh import Tanh
from .abstraction.layer import Layer
from .recurrent import Recurrent
from .dense import Dense
import numpy as np

class GRU(Layer):

    def __init__(self, units=None, forget_gate=None,
                 update_gate=None, memory_gate=None,
                 output_sequence=True, name="GRU"):

        assert forget_gate and update_gate and memory_gate or units

        self.forget_gate = (forget_gate or
                            Recurrent(input_block = Dense(units),
                                      state_block = Dense(units),
                                      activ_block = Sigmoid()))

        self.update_gate = (update_gate or
                            Recurrent(input_block = Dense(units),
                                      state_block = Dense(units),
                                      activ_block = Sigmoid()))

        self.memory_gate = (memory_gate or
                            Recurrent(input_block = Dense(units),
                                      state_block = Dense(units),
                                      activ_block = Tanh()))

        self.output_sequence = output_sequence
        self.name = name
        self.cache = []


    @property
    def blocks(self):
        yield self.forget_gate
        yield self.update_gate
        yield self.memory_gate


    def set_input_size(self, input_size):
        if input_size != self.input_size:
            self.forget_gate.set_input_size(input_size)
            self.update_gate.set_input_size(input_size)
            self.memory_gate.set_input_size(input_size)
            assert (self.forget_gate.output_size
                    == self.update_gate.output_size
                    == self.memory_gate.output_size)


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
        steps, batch_size = x.shape[0], x.shape[-1]
        state = self.init_recurrent_state(batch_size)
        outputs = [None]*steps

        for t in range(steps):
            output, state = self.step_forward(x[t], state, train)
            outputs[t] = output

        return np.array(outputs) if self.output_sequence else output


    def step_forward(self, x, state, train):
        forget, _ = self.forget_gate.step_forward(x, state, train)
        update, _ = self.update_gate.step_forward(x, state, train)
        memory, _ = self.memory_gate.step_forward(x, forget*state, train)
        next_state = update*memory + (1.0 - update)*state
        if train: self.cache.append((forget, update, memory, state))
        return next_state, next_state


    def backward(self, grad_output):
        steps, batch_size = len(self.cache), grad_output.shape[-1]
        grad_state = self.init_recurrent_gradient(batch_size)
        grad_input = [None]*steps

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
        forget, update, memory, state = self.cache.pop()
        grad_next_state = grad_output + grad_state

        grad_update = grad_next_state*(memory - state)
        grad_memory = grad_next_state*update
        grad_next_state_state = grad_next_state*(1.0 - update)

        grad_update_input, grad_update_state = (
            self.update_gate.step_backward(grad_update, 0.0))

        grad_memory_input, grad_forget_times_state = (
                self.memory_gate.step_backward(grad_memory, 0.0))

        grad_forget = grad_forget_times_state*state
        grad_memory_state = grad_forget_times_state*forget

        grad_forget_input, grad_forget_state = (
                self.forget_gate.step_backward(grad_forget, 0.0))

        grad_input = (grad_memory_input
                   +  grad_update_input
                   +  grad_forget_input)

        grad_state = (grad_next_state_state
                   +  grad_memory_state
                   +  grad_update_state
                   +  grad_forget_state)

        return grad_input, grad_state


    @property
    def is_trainable(self):
        return True


    @property
    def input_size(self):
        return self.memory_gate.input_size


    @property
    def output_size(self):
        return self.memory_gate.output_size


    @property
    def params(self):
        for block in self.blocks:
            yield from block.params


