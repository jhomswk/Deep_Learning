import numpy as np

class Exp_Moving_Average:

    def __init__(self, gain):
        self.average = 0.0
        self.gain = gain 
        self.steps = 0

    def update(self, value):
        self.steps += 1
        self.average = (self.gain*self.average
                     + (1.0 - self.gain)*value)

    def get_corrected_average(self):
        correction = 1.0 - np.power(self.gain, self.steps)
        return self.average/correction

    def reset(self):
        self.average = 0.0
        self.steps = 0


