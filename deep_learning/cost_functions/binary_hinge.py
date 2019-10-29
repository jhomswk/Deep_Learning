from .abstraction.cost_function import Cost_Function
from .hinge import Hinge
import numpy as np

class Binary_Hinge(Cost_Function):

    def __init__(self, delta=1.0):
        self.hinge = Hinge(delta)

    @property
    def delta(self):
        return self.hinge.delta

    @delta.setter
    def delta(self, value):
        self.hinge.delta = value

    def loss(self, target, prediction):
        target = self.convert(target)
        prediction = self.convert(prediction)
        return self.hinge.loss(target, prediction)

    def grad(self, target, prediction):
        target = self.convert(target)
        prediction = self.convert(prediction)
        return 2.0*self.hinge.grad(target, prediction) 

    def covert(self, labels):
        return 2.0*labels - 1.0
