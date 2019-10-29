from .abstraction.metric import Metric
import numpy as np


class Binary_Brier_Score(Metric):

    def eval(self, target, prediction):
        score = 2*np.mean(np.square(target - prediction), axis=-1)
        return score


class Categorical_Brier_Score(Metric):
    
    def eval(self, target, prediction):
        score = np.sum(np.square(target - prediction), axis=0)
        score = np.mean(score, keepdims=True)
        return score


