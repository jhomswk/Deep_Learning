from ...regularizers.null_regularizer import Null_Regularizer
from ...optimizers.null_optimizer import Null_Optimizer
from ...util.parameter import Parameter
from ...util.padder import Padder
from .layer import Layer
import numpy as np

class Convolutional(Layer):

    def __init__(self, num_kernels, kernel_size, stride, padding):

        self._num_kernels = num_kernels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._input_size = None
        self.cache = None


    def set_input_size(self, input_size):
        if self.input_size != input_size:
            self._input_size = input_size
            self._num_kernels = self._num_kernels or input_size[2]
            self._padding = Padder.get_padding(
                    self._input_size, self._kernel_size,
                    self._stride, self._padding)


    def set_optimizer(self, optimizer):
        pass


    def set_regularizer(self, regularizer):
        pass


    def forward(self, x, train):
        raise NotImplementedError


    def backward(self, grad_output):
        raise NotImplementedError


    @property
    def is_trainable(self):
        raise NotImplementedError


    @property
    def params(self):
        yield from tuple()


    def pad(self, x):
        return np.pad(x, self._padding, mode="constant")


    def unpad(self, x):
        height_start = self.top_padding
        width_start = self.left_padding
        height_end = -self.bottom_padding or None
        width_end = -self.right_padding or None
        return x[height_start:height_end, width_start:width_end]


    def extract_window(self, height_index, width_index):
        height_start = height_index*self.stride_height
        width_start = width_index*self.stride_width
        height_end = height_start + self.kernel_height
        width_end = width_start + self.kernel_width
        return (slice(height_start, height_end),
                slice(width_start, width_end))
    

    def set_cache(self, cache):
        self.cache = cache


    def get_cache(self):
        return self.cache


    @property
    def input_height(self):
        return self._input_size and self._input_size[0]


    @property
    def input_width(self):
        return self._input_size and self._input_size[1]


    @property
    def input_channels(self):
        return self._input_size and self._input_size[2]

    @property
    def input_size(self):
        return self._input_size

    @property
    def num_kernels(self):
        return self._num_kernels


    @property
    def kernel_height(self):
        return self._kernel_size[0]

    
    @property
    def kernel_width(self):
        return self._kernel_size[1]


    @property
    def stride_height(self):
        return self._stride[0]


    @property
    def stride_width(self):
        return self._stride[1]


    @property
    def padding(self):
        return self._padding


    @property
    def top_padding(self):
        return self._input_size and self._padding[0][0]


    @property
    def bottom_padding(self):
        return self._input_size and self._padding[0][1]


    @property
    def left_padding(self):
        return self._input_size and self._padding[1][0]


    @property
    def right_padding(self):
        return self._input_size and self._padding[1][1]


    @property
    def output_height(self):
        return (self._input_size and
                (1 + int((self.input_height
                  + self.top_padding + self.bottom_padding
                  - self.kernel_height)//self.stride_height)))


    @property
    def output_width(self):
        return (self._input_size and
                (1 + int((self.input_width
                  + self.left_padding + self.right_padding
                  - self.kernel_width)//self.stride_width)))


    @property
    def output_channels(self):
        return self.num_kernels


    @property
    def output_size(self):
        return (self._input_size and 
                (self.output_height,
                 self.output_width,
                 self.output_channels))


    def output_shape(self, x):
        num_obs = x.shape[-1]
        return (self.output_height,
                self.output_width,
                self.num_kernels,
                num_obs)


