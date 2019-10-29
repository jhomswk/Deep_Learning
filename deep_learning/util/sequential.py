from ..cost_functions.abstraction.cost_function import Cost_Function
from ..metrics.abstraction.metric import Metric
from ..layers.abstraction.layer import Layer

from ..cost_functions.sequential_cost import Sequential_Cost
from ..metrics.sequential_metric import Sequential_Metric
from ..layers.sequential_layer import Sequential_Layer


class Sequential:

    def __new__(self, object, *args, **kwargs):

        if isinstance(object, Layer):
            return Sequential_Layer(object, *args, **kwargs)

        if isinstance(object, Cost_Function):
            return Sequential_Cost(object, *args, **kwargs)

        if isinstance(object, Metric):
            return Sequential_Metric(object, *args, **kwargs)

