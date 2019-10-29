from .confusion_matrix import Binary_Confusion_Matrix
from .confusion_matrix import Confusion_Matrix
from .abstraction.metric import Metric
import numpy as np


class Precision(Metric):

    def eval(self, target, prediction):
        conf_mat = self.conf_mat.eval(target, prediction)

        true_positive = np.diagonal(conf_mat, axis1=-2, axis2=-1)
        false_positive = np.sum(conf_mat, axis=-2) - true_positive

        predicted_positive = true_positive + false_positive
        undefined = predicted_positive == 0
        predicted_positive[undefined] = 1.0

        score = true_positive/predicted_positive
        score[undefined] = 1.0

        return score



class Binary_Precision(Precision):
    
    def __init__(self):
        self.conf_mat = Binary_Confusion_Matrix()



class Categorical_Precision(Precision):

    def __init__(self):
        self.conf_mat = Confusion_Matrix()

