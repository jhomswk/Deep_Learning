from .abstraction.metric import Metric
from scipy.sparse import coo_matrix
import numpy as np


class Confusion_Matrix(Metric):

    def __init__(self, normalize=False):
        self.normalize = normalize

    def eval(self, target, prediction):
        target_index = np.argmax(target, axis=0)
        prediction_index = np.argmax(prediction, axis=0)
        num_classes, num_samples = prediction.shape
        confusion_matrix = coo_matrix(
                (np.ones(num_samples),
                (target_index, prediction_index)),
                shape = (num_classes, num_classes),
                dtype=np.int64).toarray()
        if self.normalize:
            confusion_matrix = confusion_matrix/num_samples
        return confusion_matrix



class Binary_Confusion_Matrix(Metric):

    def __init__(self, normalize=False):
        self.normalize = normalize

    def eval(self, target, prediction):
        prediction = np.where(prediction >= 0.5, 1, 0)
        num_classes, num_samples = prediction.shape
        confusion_matrix = np.zeros((num_classes, 2, 2))
        for cls in range(num_classes):
            confusion_matrix[cls] = coo_matrix(
                (np.ones(num_samples),
                (target[cls], prediction[cls])),
                shape = (2, 2), dtype=np.int64).toarray()
        if self.normalize:
            confusion_matrix = confusion_matrix/num_samples
        return confusion_matrix

