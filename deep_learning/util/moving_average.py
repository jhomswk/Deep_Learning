
class Moving_Average:

    def __init__(self):
        self.average = 0
        self.steps = 0

    def update(self, value):
        self.steps += 1
        self.average += (value - self.average)/self.steps

    def reset(self):
        self.average = 0
        self.steps = 0
