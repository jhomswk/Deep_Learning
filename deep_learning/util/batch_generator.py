import numpy as np

def generate_batches(x, y, batch_size=None, shuffle=True):
    num_samples = x.shape[-1]
    batch_size = batch_size or num_samples
    x, y = permute(x, y) if shuffle else (x, y)

    for i in range(0, num_samples, batch_size):
        batch = slice(i, i + batch_size)
        yield (x[..., batch], y[..., batch])


def permute(x, y):
    permutation = np.random.permutation(x.shape[-1])
    return (x[..., permutation], y[..., permutation])


