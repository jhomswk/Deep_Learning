from .confusion_matrix import Binary_Confusion_Matrix
from .confusion_matrix import Confusion_Matrix
from .abstraction.metric import Metric
import numpy as np


class Matthews(Metric):


    def __init__(self):
        raise NotImplementedError


    def eval(self, target, prediction):

        conf_mat = self.conf_mat.eval(target, prediction)

        true = np.sum(conf_mat, axis=-1)
        pred = np.sum(conf_mat, axis=-2)
        corr = np.trace(conf_mat, axis1=-2, axis2=-1)
        samp = target.shape[-1]

        num = corr*samp - np.sum(true*pred, axis=-1) 
        den = np.atleast_1d(np.sqrt(
                (np.square(samp) - np.sum(np.square(pred), axis=-1))
              * (np.square(samp) - np.sum(np.square(true), axis=-1))))
        den[den == 0] = 1.0
        score = num/den

        return score



class Binary_Matthews(Matthews):

    def __init__(self):
        self.conf_mat = Binary_Confusion_Matrix()



class Categorical_Matthews(Matthews):

    def __init__(self):
        self.conf_mat = Confusion_Matrix()

