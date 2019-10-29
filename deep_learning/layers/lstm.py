from ..activations.sigmoid import Sigmoid
from ..activations.tanh import Tanh
from .abstraction.layer import Layer
from .recurrent import Recurrent
from .dense import Dense
import numpy as np


class LSTM(Layer):


    def __init__(self, units=None, forget_gate=None,
                 update_gate=None, output_gate=None,
                 memory_gate=None, hidden_block=None,
                 output_sequence=True, name="LSTM"):

        assert (forget_gate and update_gate and
                output_gate and memory_gate or units)

        self.forget_gate = (forget_gate or
                            Recurrent(input_block = Dense(units),
                                      state_block = Dense(units),
                                      activ_block = Sigmoid()))
                
        self.update_gate = (update_gate or
                            Recurrent(input_block = Dense(units),
                                      state_block = Dense(units),
                                      activ_block = Sigmoid()))

        self.output_gate = (output_gate or 
                            Recurrent(input_block = Dense(units),
                                      state_block = Dense(units),
                                      activ_block = Sigmoid()))

        self.memory_gate = (memory_gate or
                            Recurrent(input_block = Dense(units),
                                      state_block = Dense(units),
                                      activ_block = Tanh()))

        self.hidden_block = hidden_block or Tanh()

        self.output_sequence = output_sequence
        self.name = name
        self.cache = []
    

    @property
    def blocks(self):
        yield self.forget_gate
        yield self.update_gate
        yield self.output_gate
        yield self.memory_gate
        yield self.hidden_block


    def set_input_size(self, input_size):
        if input_size != self.input_size:
            self.forget_gate.set_input_size(input_size)
            self.update_gate.set_input_size(input_size)
            self.output_gate.set_input_size(input_size)
            self.memory_gate.set_input_size(input_size)
            hidden_size = self.memory_gate.output_size
            self.hidden_block.set_input_size(hidden_size)
            assert (self.forget_gate.output_size
                    == self.update_gate.output_size
                    == self.output_gate.output_size
                    == self.memory_gate.output_size
                    == self.hidden_block.output_size)


    def set_optimizer(self, optimizer):
        for block in self.blocks:
            block.set_optimizer(optimizer)


    def set_regularizer(self, regularizer):
        for block in self.blocks:
            block.set_regularizer(regularizer)


    def init_recurrent_state(self, batch_size):
        return (np.zeros((*np.atleast_1d(self.output_size), batch_size)),
                np.zeros((*np.atleast_1d(self.output_size), batch_size)))


    def init_recurrent_gradient(self, batch_size):
        return (np.zeros((*np.atleast_1d(self.output_size), batch_size)),
                np.zeros((*np.atleast_1d(self.output_size), batch_size)))


    def forward(self, x, train):
        steps, batch_size = x.shape[0], x.shape[-1]
        state = self.init_recurrent_state(batch_size)
        outputs = [None]*steps

        for t in range(steps):
            output, state = self.step_forward(x[t], state, train)
            outputs[t] = output

        return np.array(outputs) if self.output_sequence else output


    def step_forward(self, x, state, train):
        prev_hidden, prev_memory = state

        forget, _ = self.forget_gate.step_forward(x, prev_hidden, train)
        update, _ = self.update_gate.step_forward(x, prev_hidden, train)
        output, _ = self.output_gate.step_forward(x, prev_hidden, train)
        memory, _ = self.memory_gate.step_forward(x, prev_hidden, train)

        next_memory = update*memory + forget*prev_memory
        hidden = self.hidden_block.forward(next_memory, train)
        next_hidden = output*hidden

        if train: self.cache.append(
            (forget, update, output, hidden, memory, prev_memory))

        return next_hidden, (next_hidden, next_memory)


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
        (forget, update, output,
         hidden, memory, prev_memory) = self.cache.pop()

        grad_next_hidden = grad_state[0] + grad_output

        grad_hidden = output*grad_next_hidden
        grad_output = hidden*grad_next_hidden

        grad_next_memory = grad_state[1] + (
                self.hidden_block.backward(grad_hidden))

        grad_prev_memory = grad_next_memory*forget
        grad_forget = grad_next_memory*prev_memory
        grad_memory = grad_next_memory*update
        grad_update = grad_next_memory*memory

        grad_memory_input, grad_memory_prev_hidden = (
                self.memory_gate.step_backward(grad_memory, 0.0))

        grad_output_input, grad_output_prev_hidden = (
                self.output_gate.step_backward(grad_output, 0.0))

        grad_update_input, grad_update_prev_hidden = (
                self.update_gate.step_backward(grad_update, 0.0))

        grad_forget_input, grad_forget_prev_hidden = (
                self.forget_gate.step_backward(grad_forget, 0.0))

        grad_input = (grad_memory_input
                   +  grad_output_input
                   +  grad_update_input
                   +  grad_forget_input)

        grad_prev_hidden = (grad_memory_prev_hidden
                         +  grad_output_prev_hidden
                         +  grad_update_prev_hidden
                         +  grad_forget_prev_hidden)

        return grad_input, (grad_prev_hidden, grad_prev_memory)


    @property
    def is_trainable(self):
        return True


    @property
    def input_size(self):
        return self.memory_gate.input_size

    
    @property
    def output_size(self):
        return self.hidden_block.output_size


    @property
    def params(self):
        for block in self.blocks:
            yield from block.params


