from .variance_scaler import Variance_Scaler

def He(dist, seed=None):
    return Variance_Scaler(dist=dist, mode="in", scale=2.0, seed=seed)
