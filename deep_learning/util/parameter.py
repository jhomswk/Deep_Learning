import numpy as np

class Parameter:

    def __init__(self, value=0, grad=0):
        self.value = value 
        self.grad = grad
        self.steps = 0

    def update_gradient_avg(self, gradient):
        self.steps += 1
        self.grad += (gradient - self.grad)/self.steps

    def reset_gradients(self):
        self.gradients = 0
        self.steps = 0
