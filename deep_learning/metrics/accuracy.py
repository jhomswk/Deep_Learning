from .abstraction.metric import Metric
import numpy as np


class Binary_Accuracy(Metric):

    def eval(self, target, prediction):
        prediction = np.where(prediction >= 0.5, 1, 0)
        return np.mean(target == prediction, axis=-1)


class Categorical_Accuracy(Metric):

    def eval(self, target, prediction):
        target = np.argmax(target, axis=0)
        prediction = np.argmax(prediction, axis=0)
        return np.mean(target == prediction, axis=-1, keepdims=True)

