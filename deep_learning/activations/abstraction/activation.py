from ...layers.abstraction.layer import Layer

class Activation(Layer):

    def __init__(self):
        self.cache = []
        self.input_size = None

    def setup(self, optimizer, regularizer, input_size):
        self.input_size = input_size

    def set_input_size(self, input_size):
        self.input_size = input_size

    def set_optimizer(self, optimizer):
        pass

    def set_regularizer(self, regularizer):
        pass

    def forward(self, x, train=False):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    @property
    def params(self):
        return dict()

    @property
    def output_size(self):
        return self.input_size

    @property
    def is_trainable(self):
        return False 

