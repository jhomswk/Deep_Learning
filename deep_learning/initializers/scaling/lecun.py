from .variance_scaler import Variance_Scaler

def Lecun(dist, seed=None):
    return Variance_Scaler(dist=dist, mode="in", scale=1.0, seed=seed)
