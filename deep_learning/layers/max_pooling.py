from ..regularizers.null_regularizer import Null_Regularizer
from ..optimizers.null_optimizer import Null_Optimizer
from .abstraction.convolutional import Convolutional
import numpy as np


class Max_Pooling(Convolutional):


    def __init__(self, kernel_size=(1, 1), stride=(1, 1),
                 padding="valid", input_size=None, name="Max_Pool"):
        super().__init__(None, kernel_size, stride, padding)
        self.name = name
        self.cache = []


    def forward(self, x, train):
        input = self.pad(x) 
        output = np.zeros(self.output_shape(x))
        for h in range(self.output_height):
            for w in range(self.output_width):
                window = self.extract_window(h, w)
                output[h, w] = self.max_pool(input[window])
        if train: self.cache.append(input)
        return output 


    def max_pool(self, input):
        return np.amax(input, axis=(0, 1))


    def backward(self, grad_output):
        input = self.cache[-1]
        grad_input = np.zeros(input.shape)
        for h in range(self.output_height):
            for w in range(self.output_width):
                window = self.extract_window(h, w)
                self.update_grad_input(
                        grad_input, window,
                        grad_output[h, w])
        grad_input = self.unpad(grad_input)
        del self.cache[-1]
        return grad_input


    def update_grad_input(self, grad_input, window, grad_output):
        input = self.cache[-1]
        input = input[window]
        max_mask = input == self.max_pool(input)
        grad_input[window] += max_mask*grad_output


    @property
    def is_trainable(self):
        return False

