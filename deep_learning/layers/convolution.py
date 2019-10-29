from ..optimizers.null_optimizer import Null_Optimizer
from ..regularizers.null_regularizer import Null_Regularizer
from .abstraction.convolutional import Convolutional
from ..initializers.constant.zeros import Zeros
from ..initializers.scaling.glorot import Glorot 
from ..util.moving_average import Moving_Average
from ..util.parameter import Parameter
import numpy as np

class Convolution(Convolutional):


    def __init__(self, num_kernels=1, kernel_size=(1, 1),
            stride=(1, 1), padding="valid", kernel_init = None,
            bias_init = None, reg_kernel = True, reg_bias = False,
            input_size = None, name = "Convolution"):

        super().__init__(num_kernels, kernel_size, stride, padding)

        self.regularizer = Null_Regularizer()
        self.optimizer = Null_Optimizer()

        self.kernel_init = kernel_init or Glorot("normal")
        self.kernel_grad = Moving_Average()
        self.kernel = Parameter()
        self.reg_kernel = reg_kernel

        self.bias_init = bias_init or Zeros()
        self.bias_grad = Moving_Average()
        self.bias = Parameter() 
        self.reg_bias = reg_bias

        self.name = name
        self.cache = []


    def set_input_size(self, input_size):
        if self.input_size != input_size:
            super().set_input_size(input_size)
            self.setup_parameters()


    def setup_parameters(self):
        self.kernel.value = self.kernel_init(self.kernel_shape)
        self.bias.value = self.bias_init(self.bias_shape)


    @property
    def kernel_shape(self):
            return (self.num_kernels, self.kernel_height,
                    self.kernel_width, self.input_channels)


    @property
    def bias_shape(self):
        return (self.num_kernels, 1)


    def set_optimizer(self, optimizer):
        self.optimizer.unregister(self.kernel)
        self.optimizer.unregister(self.bias)
        optimizer.register(self.kernel)
        optimizer.register(self.bias)
        self.optimizer = optimizer


    def set_regularizer(self, regularizer):
        if self.reg_kernel:
            self.regularizer.unregister(self.kernel)
            regularizer.register(self.kernel)
        if self.reg_bias:
            self.regularizer.unregister(self.bias)
            regularizer.register(self.bias)
        self.regularizer = regularizer


    def forward(self, x, train):
        input = self.pad(x) 
        output = np.zeros(self.output_shape(x))
        for h in range(self.output_height):
            for w in range(self.output_width):
                window = self.extract_window(h, w)
                output[h, w] = self.convolve(input, window)
        if train: self.cache.append(input)
        return output


    def convolve(self, input, window):
        return np.tensordot(self.kernel.value,
                            input[window],
                            axes=3) + self.bias.value
    

    def backward(self, grad_output):
        input = self.cache[-1]
        self.kernel.grad = np.zeros(self.kernel_shape)
        self.bias.grad = np.zeros(self.bias_shape)
        grad_input = np.zeros(input.shape)

        for h in range(self.output_height):
            for w in range(self.output_width):
                window = self.extract_window(h, w)
                grad_out = grad_output[h, w]
                self.update_grad_kernel(window, grad_out)
                self.update_grad_bias(window, grad_out)
                self.update_grad_input(grad_input, window, grad_out)

        grad_input = self.unpad(grad_input)
        self.kernel_grad.update(self.kernel.grad)
        self.bias_grad.update(self.bias.grad)

        del self.cache[-1]
        if len(self.cache) == 0:
            self.kernel.grad = self.kernel_grad.average
            self.bias.grad = self.bias_grad.average
            self.kernel_grad.reset()
            self.bias_grad.reset()

        return grad_input


    def update_grad_kernel(self, window, grad_output):
        input = self.cache[-1]
        batch_size = input.shape[-1]
        norm = batch_size*self.output_height*self.output_width
        self.kernel.grad += np.einsum("im, jklm -> ijkl",
                grad_output, input[window])/norm


    def update_grad_bias(self, window, grad_output):
        norm = self.output_height*self.output_width
        self.bias.grad += np.mean(grad_output, axis=-1, keepdims=True)/norm


    def update_grad_input(self, grad_input, window, grad_output):
        grad_input[window] += np.einsum("im, ijkl -> jklm",
                grad_output, self.kernel.value)/self.num_kernels


    @property
    def params(self):
        yield self.kernel
        yield self.bias


    @property
    def is_trainable(self):
        return True

