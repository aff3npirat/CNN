import numpy as np

from src.cs231n.fast_conv import im2col, col2im
from src.nodes.node import Node


class FastConv(Node):

    def __init__(self, x, w, b):
        super().__init__([x, w, b])
        self._x_col = None
        self._w_row = None

    def compute(self):
        x = self.input_nodes[0].output
        w = self.input_nodes[1].output
        b = self.input_nodes[2].output
        m, h_in, w_in, c = x.shape
        n, h_f, w_f, _ = w.shape
        h_out = h_in - h_f + 1
        w_out = w_in - w_f + 1

        self._x_col = im2col(array=x.transpose(0, 3, 1, 2), filter_dim=(h_f, w_f), stride=1)
        self._w_row = w.transpose(0, 3, 1, 2).reshape(n, -1)
        result = np.dot(self._w_row, self._x_col)
        self.output = result.reshape(n, h_out, w_out, m).transpose(3, 1, 2, 0) + b

    def backpass(self):
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        x = self.input_nodes[0].output
        w = self.input_nodes[1].output
        m, h_in, w_in, c = x.shape
        n, h_f, w_f, _ = w.shape

        for node in self.output_nodes:
            grad_cost = node.gradients[self]
            grad_cost_col = grad_cost.transpose(3, 0, 1, 2).reshape(n, -1)
            grad_cost_row = grad_cost.transpose(3, 1, 2, 0).reshape(n, -1)

            dx = np.dot(self._w_row.T, grad_cost_col)
            self.gradients[self.input_nodes[0]] += col2im(dx, (m, c, h_in, w_in), filter_dim=(h_f, w_f)).transpose(0, 2, 3, 1)
            dw = np.dot(grad_cost_row, self._x_col.T)
            self.gradients[self.input_nodes[1]] += dw.reshape(n, c, h_f, w_f).transpose(0, 2, 3, 1) / m
            self.gradients[self.input_nodes[2]] += np.sum(grad_cost, axis=(0, 1, 2)) / m
