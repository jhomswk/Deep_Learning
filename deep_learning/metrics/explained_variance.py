from .abstraction.metric import Metric
import numpy as np


class Explained_Variance(Metric):

    def eval(self, target, prediction):
        num = np.var(target - prediction, axis=-1)
        den = np.var(target, axis=-1)

        nonzero_num = num != 0
        nonzero_den = den != 0
        valid = nonzero_num & nonzero_den

        score = np.ones(target.shape[:-1])
        score[valid] = 1.0 - num[valid]/den[valid]
        score[nonzero_num & ~nonzero_den] = 0.0

        return score

