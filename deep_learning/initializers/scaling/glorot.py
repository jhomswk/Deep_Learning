from .variance_scaler import Variance_Scaler

def Glorot(dist, seed=None):
    return Variance_Scaler(dist=dist, mode="avg", scale=1.0, seed=seed)
