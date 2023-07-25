import numpy as np

from src.nodes.node import Node


def pool(x, pool_func, d_filter, stride):
    """
    Applies a pooling operation to an input array by taking HxW big subarrays and mapping each to a single value.
    Pools over all given examples (stacked along first axis) and all channels (stacked along last axis) individually.

    Args:
        x (numpy.array): Values to be pooled, must have shape MxXxYxC.
        pool_func (function(numpy.array) -> numpy.array): Function that determines pooling behaviour.  Must take one
            numpy.array with shape MxHxWxC as argument and must return a numpy.array with shape MxC.
        d_filter (list/tuple[int, int]): Size of filter in each dimension, so that d_filter= [H, W].
        stride (list/tuple[int, int]): Step size subarray is moved after applying pool_func,
            so that stride= [X_step, Y_step].

    Returns:
        Numpy.array with shape MxX_xY_xC (with X_= (X-H)//X_step + 1 and Y_= (Y-W)//Y_step + 1).
    """
    d_in = x.shape[1:]
    d_out = [len(x), (d_in[0] - d_filter[0])//stride[0] + 1, (d_in[1] - d_filter[1])//stride[1] + 1, d_in[-1]]
    out = np.zeros(d_out, dtype=x.dtype)

    y1 = out_y = 0
    y2 = d_filter[1]
    while y2 <= d_in[1]:
        x1 = out_x = 0
        x2 = d_filter[0]
        while x2 <= d_in[0]:
            x_sub = x[:, x1:x2, y1:y2]
            out[:, out_x, out_y] = pool_func(x_sub)

            x1 += stride[0]
            x2 += stride[0]
            out_x += 1
        y1 += stride[1]
        y2 += stride[1]
        out_y += 1
    return out


def deriv_pool(x, grad_cost, deriv_pool_func, d_filter, stride):
    """
    Computes the derivative of a pooling operation, which pooling operation is derived is determined by 'pool_func'.

    Args:
        x (numpy.array): Values to be used for derivation, must have shape MxXxYxC.
        grad_cost (numpy.array): Upstream gradient, must have shape MxX_xY_xC (for more information on X_, Y_ see
            'pooling.pool(...)')
        deriv_pool_func (function(numpy.array) -> numpy.array): Function that determines which pooling operation is
            derived.  Must accept numpy.array with shape MxHxWxC and must return numpy.array with same shape.
        d_filter (tuple/list[int, int]): Size of filter in each dimension, so that d_filter= [H, W].
        stride (tuple/list[int, int]): Step size subarray (after fed into deriv_pool_func) is moved,
            so that stride= [X_step, Y_step]

    Returns:
        Numpy.array with shape MxXxYxC.  Each subarray at index [i, :, :, j] is the gradient of derived pooling
        operation wrt to x[i, :, :, j]
    """
    d_in = x.shape[1:]
    out = np.zeros_like(x)

    y1 = out_y = 0
    y2 = d_filter[1]
    while y2 <= d_in[1]:
        x1 = out_x = 0
        x2 = d_filter[0]
        while x2 <= d_in[0]:
            x_sub = x[:, x1:x2, y1:y2]
            grad_cost_sub = grad_cost[:, out_x, out_y, np.newaxis, np.newaxis, :]
            out[:, x1:x2, y1:y2, :] = np.multiply(deriv_pool_func(x_sub), grad_cost_sub)

            x1 += stride[0]
            x2 += stride[0]
            out_x += 1
        y1 += stride[1]
        y2 += stride[1]
        out_y += 1
    return out


def max_pooling_func(x):
    return np.max(x, axis=(1, 2))


def mean_pooling_func(x):
    return np.mean(x, axis=(1, 2))


def deriv_max(x):
    """
    Computes gradient of max function.

    Args:
        x (numpy.array): Max function is derived wrt to x.  Must have shape MxHxWxC. Max function is then derived wrt
            each HxW big subarray of x.

    Returns:
        Numpy.array of shape MxHxWxC, so that a subarray at index [i, :, :, j] is the derivative of
        np.max(x[i, :, :, j].
    """
    gradient = np.zeros_like(x)
    max_vals = np.max(x, axis=(1, 2)).reshape(x.shape[0], 1, 1, x.shape[-1])
    gradient[x >= max_vals] = 1
    return gradient


def deriv_mean(x):
    """
    Computes gradient of mean function.

    Args:
        x (numpy.array): Mean function is derived wrt to x.  Must have shape MxHxWxC.  Mean function is then derived wrt
            each HxW big subarray of x.

    Returns:
        Numpy.array of shape MxHxWxC, so that a subarray at index [i, :, :, j] is the derivative of
        np.mean(x[i, :, :, j].
    """
    return np.full_like(x, 1/np.prod(x.shape[1:-1]))


class MaxPool(Node):
    """
    Node that applies max pooling to input.

    Attributes:
        d_filter (list/tuple[int, int]): Size of filter used for pooling.
        stride (list/tuple[int, int]): Sie of step filter is moved.
    """

    def __init__(self, x, d_filter=(2, 2), stride=(2, 2)):
        """
        Args:
            x (Node): Values max pooling is applied to.
            d_filter (list/tuple[int, int]): Size of filter used for pooling, must be [H, W].
            stride (list/tuple[int, int]): Step size each filter is moved, must be [X_step, Y_step].
        """
        super().__init__([x])
        self.d_filter = d_filter
        self.stride = stride

    def compute(self):
        """
        Applies max-pooling to x.output.
        """
        x = self.input_nodes[0].output
        self.output = pool(x, max_pooling_func, self.d_filter, self.stride)

    def backpass(self):
        """
        Computes gradients of graph-output wrt x.
        """
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        x = self.input_nodes[0].output
        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            self.gradients[self.input_nodes[0]] += deriv_pool(x, grad_cost, deriv_max, self.d_filter, self.stride)


class MeanPool(Node):
    """
    Node that applies mean pooling to input.

    Attributes:
        d_filter (list/tuple[int, int]): See MaxPool class.
        stride (list/tuple[int, int]): See MaxPool class.
    """

    def __init__(self, x, d_filter=(2, 2), stride=(2, 2)):
        """
        See MaxPool class.
        """
        super().__init__([x])
        self.d_filter = d_filter
        self.stride = stride

    def compute(self):
        """
        Applies mean-pooling operation to x.output.
        """
        x = self.input_nodes[0].output
        self.output = pool(x, mean_pooling_func, self.d_filter, self.stride)

    def backpass(self):
        """
        Computes gradient of graph-output wrt x.
        """
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        x = self.input_nodes[0].output

        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            self.gradients[self.input_nodes[0]] += deriv_pool(x, grad_cost, deriv_mean, self.d_filter, self.stride)
