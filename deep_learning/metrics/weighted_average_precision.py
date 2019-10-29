from .binary_classification_curve import unique_indices
from .precision_recall_curve import precision_recall_curve
from .abstraction.metric import Metric
import numpy as np


class Weighted_Average_Precision(Metric):

    def eval(self, target, prediction):

        num_classes = prediction.shape[0]
        score = np.zeros(num_classes)
        for cls in range(num_classes):

            precision, recall, _ = precision_recall_curve(
                        target[cls], prediction[cls])

            reverse = slice(None, None, -1)
            precision = precision[reverse]
            recall = recall[reverse]

            indices = unique_indices(recall)
            precision = precision[indices]
            recall = recall[indices]

            score[cls] = -np.sum(np.diff(recall)*precision[:-1])

        return score



