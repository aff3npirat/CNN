import numpy as np

from src.nodes.node import Node


def convolve(x, w, b=None):
    """
    Convolves N kernels over M given examples.  Each example has shape (h_in, w_in, c).

    Args:
         x (numpy.array): Input values, must have shape (m, h_in, w_in, c).
         w (numpy.array): Kernel weights, must have shape (n, h_f, w_f, c).
         b (numpy.array): Optional; Bias used, if not none must have shape (n,).

    Returns:
        Numpy.array with shape MxX_xY_xN (with X_= X-H+1 and Y_= Y-W+1), so that subarray at index [i, :, :, j] is the
        feat map of jth kernel convolving over ith example.
    """
    m, h_in, w_in, c = x.shape
    n, h_f, w_f, c = w.shape
    d_out = [m, h_in - h_f + 1, w_in - w_f + 1, n]
    out = np.zeros(d_out)

    for yy in range(d_out[2]):
        for xx in range(d_out[1]):
            x_sub = x[:, np.newaxis, xx:xx+h_f, yy:yy+w_f, :]
            out[:, xx, yy, :] = np.sum(np.multiply(w, x_sub), axis=(2, 3, 4))
            if b is not None:
                out[:, xx, yy] += b
    return out


def convolve_iterative(x, w, b=None):
    """
    See convolve(x, w, b=None).
    """
    m, h_in, w_in, c = x.shape
    n, h_w, w_w, _ = w.shape
    d_out = [m, h_in - h_w + 1, w_in - w_w + 1, n]
    out = np.zeros(d_out, dtype=x.dtype)

    for f in range(n):
        for yy in range(d_out[2]):
            for xx in range(d_out[1]):
                for v in range(m):
                    for d in range(c):
                        x_sub = x[v, xx:xx+h_w, yy:yy+w_w, d]
                        out[v, xx, yy, f] += np.sum(np.multiply(x_sub, w[f, :, :, d]))
                    if b is not None:
                        out[v, xx, yy, f] += b[f]
    return out


def apply_zero_padding(x, p):
    """
    Adds zeros to the edges of each subarray of shape XxY, so that after a convolution operation the resulting
        featmaps all have shape XxY.

    Args:
        x (numpy.array): Values to be padded, must have shape (m, h_in, w_in, c).
        p (list/tuple[int, int]): Amount of zeros added to edges.

    Returns:
        Numpy.array with shape (m, h_out, w_out, c) (with h_out= h_in + 2*p[0] and w_out= w_in + 2*p[1]).
    """
    m, h_in, w_in, c = x.shape
    d_padded = (m, h_in + 2*p[0], w_in + 2*p[1], c)
    padded = np.zeros(d_padded, dtype=x.dtype)
    padded[:, p[0]:d_padded[1] - p[0], p[1]:d_padded[2] - p[1], :] = x
    return padded


class ZeroPadding(Node):
    """
    Node that applies zero padding.

    Adds zeros to the edges of input.  Input should be numpy.array of shape MxXxYxC, zeros are added so that X, Y axis
    size increases along X, Y axes.

    Attributes:
        p (tuple/list[int, int]): Amount of zeros added to a single edge.  First element indicates amount of zeros added
            to both edges along X axis and second element to edges along Y axis.
    """

    def __init__(self, x, p):
        """
        Args:
            x (Node): Node storing input, x.output should be numpy.array with shape MxXxYxC
            p (tuple/list[int, int]): Amount of zeros added to each edge.
        """
        super().__init__([x])
        self.p = p

    def compute(self):
        x = self.input_nodes[0].output
        self.output = apply_zero_padding(x, self.p)

    def backpass(self):
        """
        See base class.
        """
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        _, h_in, w_in, _ = self.input_nodes[0].output.shape
        p = self.p
        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            self.gradients[self.input_nodes[0]] += grad_cost[:, p[0]:h_in + p[0], p[1]:w_in + p[1], :]


class Convolution(Node):
    """
    Node that represents the convolution operation.
    """

    def __init__(self, x, w, b):
        """
        Args:
            x (Node): Node storing input., x.output should be numpy.array with shape MxXxYxC.
            w (Node): Node storing weights, w.output should be numpy.array with shape NxHxWxC.
            b (Node): Node storing bias, b.output should be numpy.array with shape N.
        """
        super().__init__([x, w, b])

    def compute(self):
        """
        Convolve over image.

        Moves each filter over all filter-sized subimages of input.  Each subimage is mapped to one output value
        so that resulting image consists of 2D images (called feature map) stacked along last axis.
        """
        x = self.input_nodes[0].output
        w = self.input_nodes[1].output
        b = self.input_nodes[2].output

        self.output = convolve(x, w, b)

    def backpass(self):
        """
        Computes gradient of graph-output (L) wrt x, w, b.

        dL/dx equals a full convolution over upstream gradient and kernels rotatet by
        180 degree.  dL/dw equals a convolution over x and upstream gradient.  dL/db_i (bias of ith kernel)
        equals sum of ith upstream gradient (gradient wrt ith feat map).
        """
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        x = self.input_nodes[0].output
        m = len(x)
        w = self.input_nodes[1].output
        p = [w.shape[1] - 1, w.shape[2] - 1]
        w_rot180 = np.rot90(w, 2, axes=(1, 2))

        for node in self.output_nodes:
            grad_cost = node.gradients[self]
            grad_cost_pad = apply_zero_padding(grad_cost, p)

            self.gradients[self.input_nodes[0]] += convolve(grad_cost_pad, w_rot180.transpose(3, 1, 2, 0))
            dw = convolve(x.transpose(3, 1, 2, 0), grad_cost.transpose(3, 1, 2, 0)).transpose(3, 1, 2, 0)
            self.gradients[self.input_nodes[1]] += dw / m
            self.gradients[self.input_nodes[2]] += np.sum(grad_cost, axis=(0, 1, 2)) / m


