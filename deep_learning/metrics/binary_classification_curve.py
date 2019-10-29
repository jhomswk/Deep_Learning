import numpy as np

def binary_classification_curve(target, prediction):

    target = np.squeeze(target)
    prediction = np.squeeze(prediction)

    assert target.ndim == 1
    assert prediction.ndim == 1

    target = target.ravel()
    prediction = prediction.ravel()

    indices = np.argsort(prediction)[::-1]
    target = target[indices]
    prediction = prediction[indices]

    indices = unique_indices(prediction)

    true_positive = np.cumsum(target)[indices]
    false_positive = 1 + indices - true_positive
    threshold = prediction[indices]

    indices = not_collinear_indices(
            true_positive, false_positive)

    true_positive = true_positive[indices]
    false_positive = false_positive[indices]
    threshold = threshold[indices]

    true_positive = prepend(0, true_positive)
    false_positive = prepend(0, false_positive)
    threshold = prepend(1.0, threshold)

    return false_positive, true_positive, threshold


def unique_indices(x):
    unique = np.diff(x, axis=-1)
    unique = append(True, unique)
    unique = np.nonzero(unique)[0]
    return unique


def append(value, x):
    return np.r_[x, value]
    

def not_collinear_indices(x, y):
    not_collinear = np.logical_or(
            np.diff(x, 2, axis=-1),
            np.diff(y, 2, axis=-1))
    not_collinear = enclose(True, not_collinear)
    not_collinear = np.nonzero(not_collinear)[0]
    return not_collinear


def enclose(value, x):
    return np.r_[value, x, value]


def prepend(value, x):
    return np.r_[value, x]


