import numpy as np

def sequenciate(function, *args, **kwargs):
    return np.array(list(map(function, *args, **kwargs)))
