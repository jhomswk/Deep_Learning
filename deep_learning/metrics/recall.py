from .confusion_matrix import Binary_Confusion_Matrix
from .confusion_matrix import Confusion_Matrix
from .abstraction.metric import Metric
import numpy as np


class Recall(Metric):

    def eval(self, target, prediction):
        conf_mat = self.conf_mat.eval(target, prediction)

        true_positive = np.diagonal(conf_mat, axis1=-2, axis2=-1)
        false_negative = np.sum(conf_mat, axis=-1) - true_positive

        positive = true_positive + false_negative
        undefined = positive == 0
        positive[undefined] = 1.0

        score = true_positive/positive
        score[undefined] = 1.0

        return score



class Binary_Recall(Recall):
    
    def __init__(self):
        self.conf_mat = Binary_Confusion_Matrix()



class Categorical_Recall(Recall):

    def __init__(self):
        self.conf_mat = Confusion_Matrix()

