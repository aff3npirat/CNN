import numpy as np


def generate_batches(arr, batch_size):
    """
    Splits input into multiple arrays of equal size (called batches).

    Args:
        arr (numpy.array): Array to split with shape (n, ...).
        batch_size (int): Size of each batch.

    Returns:
        List with shape (n/bs, ...) so that ith element is arr[i*bs:(i+1)*bs] (bs= batch_size).
    """
    batches = []
    for i in range(0, arr.shape[0], batch_size):
        batches.append(arr[i:min(i + batch_size, arr.shape[0])])
    return batches


def categorical2one_hot(y, d_y):
    one_hot_mat = np.zeros((len(y), d_y), dtype=float)
    one_hot_mat[(np.arange(len(y)), y)] = 1.0
    return one_hot_mat

