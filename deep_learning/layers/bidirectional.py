from .abstraction.layer import Layer
import numpy as np


class Bidirectional(Layer):


    def __init__(self, forward_block, backward_block,
                 mode="concat", name="Bidirectional"):

        assert (forward_block.output_sequence
                == backward_block.output_sequence)

        self.fwd_block = forward_block
        self.bwd_block = backward_block
        self.mode = mode
        self.name = name
        self.cache = []


    @property
    def blocks(self):
        yield self.fwd_block
        yield self.bwd_block

    
    def set_input_size(self, input_size):
        if input_size != self.input_size:
            self.fwd_block.set_input_size(input_size)
            self.bwd_block.set_input_size(input_size)
            fwd_out_size = np.atleast_1d(self.fwd_block.output_size)
            bwd_out_size = np.atleast_1d(self.bwd_block.output_size)
            last = -1 if self.mode == "concat" else None
            assert np.all(fwd_out_size[:last] == bwd_out_size[:last])


    def set_optimizer(self, optimizer):
        for block in self.blocks:
            block.set_optimizer(optimizer)


    def set_regularizer(self, regularizer):
        for block in self.blocks:
            block.set_regularizer(regularizer)


    def forward(self, x, train):

        fwd_output = self.fwd_block.forward(x, train)
        bwd_output = self.bwd_block.forward(x[::-1], train)

        if self.bwd_block.output_sequence:
            bwd_output = bwd_output[::-1]

        output = self.merge(fwd_output, bwd_output)

        if train: self.cache.append((fwd_output, bwd_output))

        return output


    def merge(self, forward, backward):
        if self.mode == "concat":
            return np.r_["-2", forward, backward]

        if self.mode == "sum":
            return forward + backward

        if self.mode == "avg":
            return (forward + backward)/2.0

        if self.mode == "mul":
            return forward*backward


    def backward(self, grad_output):

        grad_fwd_output, grad_bwd_output = (
                self.merge_backward(grad_output))

        if self.bwd_block.output_sequence:
            grad_bwd_output = grad_bwd_output[::-1]

        grad_bwd_input = self.bwd_block.backward(grad_bwd_output)[::-1]
        grad_fwd_input = self.fwd_block.backward(grad_fwd_output)

        grad_input = 0.5*(grad_fwd_input + grad_bwd_input)

        return grad_input


    def merge_backward(self, grad_output):
        fwd_output, bwd_output = self.cache.pop()

        if self.mode == "concat":
            fwd_size = np.atleast_1d(self.fwd_block.output_size)[-1]
            grad_fwd_output = grad_output[..., :fwd_size, :]
            grad_bwd_output = grad_output[..., fwd_size:, :] 

        elif self.mode == "sum":
            grad_fwd_output = grad_bwd_output = grad_output

        elif self.mode == "avg":
            grad_fwd_output = grad_bwd_output = 0.5*grad_output

        elif self.mode == "mul":
            grad_fwd_output = grad_output*bwd_output
            grad_bwd_output = grad_output*fwd_output

        return (grad_fwd_output, grad_bwd_output)


    @property
    def is_trainable(self):
        return any(map(lambda x: x.is_trainable, self.blocks))


    @property
    def input_size(self):
        return self.fwd_block.input_size

    
    @property
    def output_size(self):
        fwd_out_size = np.atleast_1d(self.fwd_block.output_size)
        bwd_out_size = np.atleast_1d(self.bwd_block.output_size)
        output_size = fwd_out_size[:]

        if output_size[0] is None:
            return None

        if self.mode == "concat":
            output_size[-1] += bwd_out_size[-1]

        if output_size.ndim == 1:
            output_size = output_size[0]

        return output_size


    @property
    def params(self):
        for block in self.blocks:
            yield from block.params


