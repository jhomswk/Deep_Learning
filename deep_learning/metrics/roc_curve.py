from .binary_classification_curve import binary_classification_curve
import numpy as np

def roc_curve(target, prediction):

    false_positive, true_positive, threshold = (
            binary_classification_curve(target, prediction))

    positive = true_positive[-1] or np.inf
    negative = false_positive[-1] or np.inf
    true_positive = true_positive/positive
    false_positive = false_positive/negative

    return false_positive, true_positive, threshold
