from ..regularizers.null_regularizer import Null_Regularizer
from ..optimizers.null_optimizer import Null_Optimizer
from ..initializers.constant.ones import Ones
from ..initializers.constant.zeros import Zeros
from ..util.moving_average import Moving_Average
from ..util.exp_moving_average import Exp_Moving_Average
from ..util.parameter import Parameter
from .abstraction.layer import Layer
import numpy as np

class Batch_Norm(Layer):

    def __init__(self, gamma_init=None, beta_init=None, reg_gamma=True,
                 reg_beta=False, mean_init=None, var_init=None,
                 momentum=0.9, epsilon=0.001, name="Batch_Norm"):

        self.gamma = Parameter()
        self.beta = Parameter()
        self.gamma_grad = Moving_Average()
        self.beta_grad = Moving_Average()
        self.mean = Exp_Moving_Average(momentum)
        self.var = Exp_Moving_Average(momentum) 

        self.gamma_init = gamma_init or Ones()
        self.beta_init = beta_init or Zeros()
        self.mean_init = mean_init or Zeros()
        self.var_init = var_init or Ones()
        self.momentum = momentum
        self.epsilon = epsilon

        self.reg_gamma = reg_gamma
        self.reg_beta = reg_beta

        self.regularizer = Null_Regularizer()
        self.optimizer = Null_Optimizer() 
        self.num_features = None
        self.input_size = None
        self.axes = None
        self.name = name
        self.cache = []


    def set_input_size(self, input_size):
        if self.input_size != input_size:
            self.input_size = input_size
            self.set_batch_norm_mode()
            self.setup_parameters()


    def set_batch_norm_mode(self):
        if isinstance(self.input_size, (tuple, list, np.ndarray)):
            self.set_spatial_batch_norm_mode()
        else:
            self.set_local_batch_norm_mode()

    
    def set_spatial_batch_norm_mode(self):
        self.num_features = self.input_size[2]
        self.axes = (0, 1, 3)


    def set_local_batch_norm_mode(self):
        self.num_features = self.input_size
        self.axes = -1


    def setup_parameters(self):
        shape = (self.num_features, 1)
        self.gamma.value = self.gamma_init(shape)
        self.beta.value = self.beta_init(shape)
        self.mean.average = self.mean_init(shape)
        self.var.average = self.var_init(shape)


    def set_optimizer(self, optimizer):
        self.optimizer.unregister(self.gamma)
        self.optimizer.unregister(self.beta)
        optimizer.register(self.gamma)
        optimizer.register(self.beta)
        self.optimizer = optimizer


    def set_regularizer(self, regularizer):
        if self.reg_gamma:
            self.regularizer.unregister(self.gamma)
            regularizer.register(self.gamma)
        if self.reg_beta:
            self.regularizer.unregister(self.beta)
            regularizer.register(self.beta)
        self.regularizer = regularizer


    def forward(self, x, train):
        if train:
            mean = np.mean(x, axis=self.axes, keepdims=True)
            var = np.var(x, axis=self.axes, keepdims=True)
            x_hat = self.standarize(x, mean, var) 
            self.mean.update(mean)
            self.var.update(var)
            self.cache.append((x_hat, self.var.average))
            return self.scale(x_hat)

        return self.scale(self.standarize(
                    x, self.mean.average, self.var.average))


    def standarize(self, x, mean, var):
        return (x - mean)/np.sqrt(var + self.epsilon)


    def scale(self, x):
        return self.gamma.value*x + self.beta.value


    def backward(self, grad_output):
        self.gamma.grad = self.compute_grad_gamma(grad_output)
        self.beta.grad = self.compute_grad_beta(grad_output)
        grad_input = self.compute_grad_input(grad_output)

        self.gamma_grad.update(self.gamma.grad)
        self.beta_grad.update(self.beta.grad)

        del self.cache[-1]

        if len(self.cache) == 0:
            self.gamma.grad = self.gamma_grad.average
            self.beta.grad = self.beta_grad.average
            self.gamma_grad.reset()
            self.beta_grad.reset()

        return grad_input 


    def compute_grad_gamma(self, grad_output):
        x_hat, _ = self.cache[-1]
        return np.mean(grad_output*x_hat, axis=self.axes, keepdims=True)


    def compute_grad_beta(self, grad_output):
        return np.mean(grad_output, axis=self.axes, keepdims=True) 


    def compute_grad_input(self, grad_output):
        x_hat, var = self.cache[-1]
        grad_input = grad_output - self.beta.grad - x_hat*self.gamma.grad
        grad_input *= self.gamma.value/np.sqrt(var + self.epsilon)
        return grad_input


    @property
    def params(self):
        yield self.gamma
        yield self.beta


    @property
    def output_size(self):
        return self.input_size


    @property
    def is_trainable(self):
        return True
    
