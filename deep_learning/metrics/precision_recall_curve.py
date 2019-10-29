from .binary_classification_curve import binary_classification_curve
import numpy as np

def precision_recall_curve(target, prediction):

    false_positive, true_positive, threshold = (
            binary_classification_curve(target, prediction))

    precision = true_positive + false_positive
    precision = precision.astype(np.float64)
    precision[precision == 0] = np.inf
    precision = true_positive/precision

    positive = true_positive[-1] or np.inf
    recall = true_positive/positive

    return precision, recall, threshold
