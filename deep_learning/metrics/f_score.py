from .confusion_matrix import Binary_Confusion_Matrix
from .confusion_matrix import Confusion_Matrix
from .abstraction.metric import Metric
import numpy as np


class F_Score(Metric):

    def __init__(self, beta=1.0):
        raise NotImplementedError

    def eval(self, target, prediction):
        conf_mat = self.conf_mat.eval(target, prediction)
        true_positive  = np.diagonal(conf_mat, axis1=-2, axis2=-1) 
        false_positive = np.sum(conf_mat, axis=-2) - true_positive
        false_negative = np.sum(conf_mat, axis=-1) - true_positive 

        beta_square = np.square(self.beta)
        num = (1.0 + beta_square)*true_positive
        denom = num + false_positive + beta_square*false_negative
        undefined = denom == 0
        denom[undefined] = 1.0

        score = num/denom
        score[undefined] = 1.0
        return score


class Binary_F_Score(F_Score):

    def __init__(self, beta=1.0):
        self.conf_mat = Binary_Confusion_Matrix()
        self.beta = beta


class Categorical_F_Score(F_Score):

    def __init__(self, beta=1.0):
        self.conf_mat = Confusion_Matrix()
        self.beta = beta

