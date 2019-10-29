from .abstraction.metric import Metric
from ..util.sequenciate import sequenciate
import numpy as np

class Sequential_Metric(Metric):

    def __init__(self, metric):
        self.metric = metric

    def eval(self, target, prediction):
        return sequenciate(self.metric.eval, target, prediction)

