
class Score:

    def set_input_size(self, input_size, state_size):
        pass

    def set_optimizer(self, optimizer):
        pass

    def set_regularizer(self, regularizer):
        pass

    def forward(self, x, state, train):
        raise NotImplementedError

    def backward(self, grad_output, grad_state):
        raise NotImplementedError

    @property
    def params(self):
        yield from tuple()
