import numpy as np

class Moving_Max_Norm:

    def __init__(self, gain):
        self.gain = gain
        self.norm = 0.0

    def update(self, value):
        self.norm = np.maximum(
              self.gain*self.norm, np.abs(value))

    def reset(self):
        self.norm = 0.0
