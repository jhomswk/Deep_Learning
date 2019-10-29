from .sequential_layer import Sequential_Layer as Sequential
from ..util.moving_average import Moving_Average
from ..activations.softmax import Softmax
from ..activations.tanh import Tanh
from .abstraction.layer import Layer
from .dense import Dense
from .block import Block
import numpy as np


class Attention(Layer):


    def __init__(self, recurrence, score, name=None):
        self.name = name or f"Attention[{recurrence.name}]"
        self.grad_input = Moving_Average()
        self.recurrence = recurrence
        self.softmax = Softmax()
        self.score = score
        self.cache = []


    @property
    def blocks(self):
        yield self.recurrence
        yield self.softmax
        yield self.score


    def set_input_size(self, input_size):
        if input_size != self.input_size:
            self.recurrence.set_input_size(input_size)
            state_size = self.recurrence.output_size
            self.score.set_input_size(input_size, state_size)


    def set_optimizer(self, optimizer):
        for block in self.blocks:
            block.set_optimizer(optimizer)


    def set_regularizer(self, regularizer):
        for block in self.blocks:
            block.set_regularizer(regularizer)


    def init_recurrent_state(self, batch_size):
        return self.recurrence.init_recurrent_state(batch_size)


    def init_recurrent_gradient(self, batch_size):
        return self.recurrence.init_recurrent_gradient(batch_size)


    def forward(self, x, train):
        steps, batch_size = x.shape[0], x.shape[-1]
        state = self.init_recurrent_state(batch_size)
        outputs = [None]*steps

        for t in range(steps):
            outputs[t], state = self.step_forward(x, state, train)

        return np.array(outputs) if self.output_sequence else outputs[-1]


    def step_forward(self, x, state, train):
        scores = self.score.forward(x, state, train)
        alphas = self.softmax.forward(scores, train)
        context = np.sum(alphas*x, axis=0)
        output, state = self.recurrence.step_forward(context, state, train)
        if train: self.cache.append((x, alphas))
        return output, state


    def backward(self, grad_output):
        steps, batch_size = len(self.cache), grad_output.shape[-1]
        grad_state = self.init_recurrent_gradient(batch_size)
        self.grad_input.reset()

        if not self.output_sequence:
            grad_out = [0]*steps
            grad_out[-1] = grad_output
            grad_output = grad_out

        for t in reversed(range(steps)):
            grad_input, grad_state = (
                    self.step_backward(grad_output[t], grad_state))
            self.grad_input.update(grad_input)

        grad_input = self.grad_input.average
        return grad_input 


    def step_backward(self, grad_output, grad_state):
        input, alphas = self.cache.pop()

        grad_context, grad_state = (
            self.recurrence.step_backward(grad_output, grad_state))

        grad_input = grad_context*alphas
        grad_alphas = np.mean(grad_context*input, axis=-2, keepdims=True)
        grad_score = self.softmax.backward(grad_alphas)

        grad_score_input, grad_score_state = (
                self.score.backward(grad_score))

        grad_state = grad_state + grad_score_state
        grad_input = grad_input + grad_score_input

        return grad_input, grad_state


    @property
    def is_trainable(self):
        return any(block.is_trainable for block in self.blocks)


    @property
    def input_size(self):
        return self.recurrence.input_size


    @property
    def output_size(self):
        return self.recurrence.output_size


    @property
    def output_sequence(self):
        return self.recurrence.output_sequence


    @property
    def params(self):
        for block in self.blocks:
            yield from block.params


