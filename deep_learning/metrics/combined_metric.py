from .abstraction.metric import Metric
import numpy as np


class Combined_Metric(Metric):


    def __init__(self, metrics):
        self.metrics = metrics


    def eval(self, target, prediction):
        return np.array(list(self.scores(target, prediction)))


    def scores(self, target, prediction):
        for metric in self.metrics:
            score = metric.eval(target, prediction)
            score = np.reshape(score, -1)
            yield from score

