from .abstraction.metric import Metric

class Null_Metric(Metric):

    def eval(self, target, prediction):
        return 0 

