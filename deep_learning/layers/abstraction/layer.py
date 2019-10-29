
class Layer(object):

    def set_input_size(self, input_size):
        raise NotImplementedError

    def set_optimizer(self, optimizer):
        pass

    def set_regularizer(self, regularizer):
        pass

    def forward(self, input, train):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    @property
    def is_trainable(self):
        raise NotImplementedError

    @property
    def output_size(self):
        raise NotImplementedError
        
    @property
    def params(self):
        yield from tuple()


