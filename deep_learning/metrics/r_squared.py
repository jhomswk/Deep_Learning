from .abstraction.metric import Metric
import numpy as np


class Rsquared(Metric):

    def eval(self, target, prediction):
        target_mean = np.mean(target, axis=-1, keepdims=True)
        num = np.sum(np.square(target - prediction), axis=-1)
        den = np.sum(np.square(target - target_mean), axis=-1)

        nonzero_den = den != 0
        nonzero_num = num != 0
        valid = nonzero_num & nonzero_den

        score = np.ones(prediction.shape[:-1])
        score[valid] = 1.0 - num[valid]/den[valid]
        score[nonzero_num & ~nonzero_den] = 0.0

        return score 
