from .abstraction.metric import Metric
from .roc_curve import roc_curve 
import numpy as np

class ROC_AUC(Metric):

    def eval(self, target, prediction):

        num_units = prediction.shape[0]
        au_roc = np.zeros(num_units)

        for out_unit in range(num_units):

            false_positive_rate, true_positive_rate, _ = roc_curve(
                    target[out_unit], prediction[out_unit])

            au_roc[out_unit] = np.trapz(
                    y=true_positive_rate, x=false_positive_rate)

        return au_roc 

