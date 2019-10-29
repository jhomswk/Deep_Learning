from ..cost_functions.null_cost_function import Null_Cost_Function
from ..regularizers.null_regularizer import Null_Regularizer
from ..optimizers.null_optimizer import Null_Optimizer
from ..metrics.null_metric import Null_Metric
from .tracker.tracker import Tracker
from ..layers.block import Block
from ..util.batch_generator import generate_batches
import matplotlib.pyplot as plt


class Neural_Network:


    def __init__(self):
        self.block = Block()
        self.set_cost_function(Null_Cost_Function())
        self.set_metric_function(Null_Metric())
        self.set_regularizer(Null_Regularizer())
        self.set_optimizer(Null_Optimizer())
        self.tracker = Tracker(self)


    @property
    def layers(self):
        return self.block.layers


    def set_cost_function(self, cost_function):
        self.cost_function = cost_function


    def set_metric_function(self, metric_function):
        self.metric_function = metric_function


    @property
    def regularizer(self):
        return self.block.regularizer


    def set_regularizer(self, regularizer):
        regularizer = regularizer or Null_Regularizer()
        self.block.set_regularizer(regularizer)


    @property
    def optimizer(self):
        return self.block.optimizer


    def set_optimizer(self, optimizer):
        optimizer = optimizer or Null_Optimizer()
        self.block.set_optimizer(optimizer)


    def set_input_size(self, input_size):
        self.block.set_input_size(input_size)


    def add(self, layer):
        self.block.add(layer)


    def train(self, x_train, y_train, x_valid=None, y_valid=None, 
              epochs=1000, batch_size=None, shuffle=True, plot=False):
        
        self.tracker.set_training_set(x_train, y_train)
        self.tracker.set_validation_set(x_valid, y_valid)

        for epoch in range(epochs):
            self.tracker.start_new_epoch(epoch, epochs)

            for input, target in generate_batches(
                    x_train, y_train, batch_size, shuffle):

                prediction, _ = self.train_step(input, target)
                self.tracker.track(input, target, prediction)

            self.tracker.finish_current_epoch()

        self.tracker.show_tracking()

        if plot:
            self.plot_tracking()


    def train_step(self, input, target):
        prediction = self.forward(input, train=True)
        cost_gradient = self.compute_cost_gradient(target, prediction)
        input_gradient = self.backward(cost_gradient)
        self.regularizer.regularize_params()
        self.optimizer.update_params()
        return prediction, input_gradient


    def forward(self, x, train):
        return self.block.forward(x, train)


    def backward(self, grad_output):
        return self.block.backward(grad_output)


    def compute_cost(self, target, prediction):
        cost = self.cost_function.eval(target, prediction)
        cost += self.regularizer.regularize_cost()
        return cost


    def compute_cost_gradient(self, target, prediction):
        return self.cost_function.grad(target, prediction)


    def compute_metric(self, target, prediction):
        return self.metric_function.eval(target, prediction)


    def set_tracking_rate(self, rate):
        self.tracker.set_tracking_rate(rate)


    def set_showing_rate(self, rate):
        self.tracker.set_showing_rate(rate)


    def show_tracking(self, epoch=None):
        self.tracker.show_tracking(self, epoch)


    def plot_tracking(self):
       plt.show(self.tracker.plot_tracking())


    def reset_tracking(self):
        self.tracker.reset_tracking()
        

    def predict(self, x):
        return self.forward(x, train=False)


    @property
    def input_size(self):
        return self.block.input_size


    @property
    def output_size(self):
        return self.block.output_size


    @property
    def params(self):
        yield from self.block.params


