import numpy as np
import unittest

from src.nodes import placeholders
from src.nodes.operations import pooling


class TestMaxPool(unittest.TestCase):

    def test_compute_with_even_size_even_stride(self):
        # given
        x = placeholders.Variable(np.arange(30720).reshape(10, 32, 32, 3) - 1000)
        node = pooling.MaxPool(x)

        # when
        node.compute()

        # then
        assert node.output.shape == (10, 16, 16, 3)
        # np.max(x.output[m, i:i+x, j:j+y, c])
        # np.max() always returns element at [m, i+x-1, j+y-1, c], due to np.arange().
        assert node.output[0, 2, 4, 0] == x.output[0, 5, 9, 0]
        assert node.output[4, 15, 15, 1] == x.output[4, 31, 31, 1]

    def test_compute_with_odd_size_even_stride(self):
        # given
        x = placeholders.Variable(np.arange(5070).reshape(10, 13, 13, 3))
        node = pooling.MaxPool(x)

        # when
        node.compute()

        # then
        assert node.output.shape == (10, 6, 6, 3)
        assert node.output[9, 5, 5, 2] == x.output[9, 11, 11, 2]

    def test_compute_with_odd_size_odd_stride(self):
        # given
        x = placeholders.Variable(np.arange(5070).reshape(10, 13, 13, 3))
        node = pooling.MaxPool(x, stride=(3, 3), d_filter=(3, 3))

        # when
        node.compute()

        # then
        assert node.output.shape == (10, 4, 4, 3)
        assert node.output[0, 3, 3, 0] == x.output[0, 11, 11, 0]

    def test_compute_with_even_size_odd_stride(self):
        # given
        x = placeholders.Variable(np.arange(30720).reshape(10, 32, 32, 3))
        node = pooling.MaxPool(x, stride=(3, 3), d_filter=(3, 3))

        # when
        node.compute()

        # then
        assert node.output.shape == (10, 10, 10, 3)
        assert node.output[0, 9, 9, 0] == x.output[0, 29, 29, 0]

    def test_backpass_with_upstream_gradient(self):
        # given
        x = placeholders.Variable(np.arange(30720).reshape(10, 32, 32, 3))
        node = pooling.MaxPool(x)

        # when
        upstream_grad = np.arange(7680).reshape(10, 16, 16, 3) + 1
        placeholders.GradPlaceholder(node).backpass(upstream_grad)
        node.backpass()

        # then
        assert node.gradients[x].shape == (10, 32, 32, 3)
        assert node.gradients[x][0, 1, 1, 0] == upstream_grad[0, 0, 0, 0]