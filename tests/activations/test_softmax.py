import numpy as np
import unittest

from src.nodes import placeholders
from src.nodes.activations import softmax


class TestSoftmax(unittest.TestCase):

    def test_compute_with_positive_input(self):
        # given
        x = placeholders.Variable(np.arange(12).reshape(3, 4))

        # when
        node = softmax.Softmax(x)
        node.compute()

        # then
        assert node.output.sum(axis=1).sum() == 3
        assert (node.output == [[0.03, 0.09, 0.24, 0.64], [0.03, 0.09, 0.24, 0.64], [0.03, 0.09, 0.24, 0.64]]).all()

    def test_compute_with_negative_input(self):
        # given
        x = placeholders.Variable(-np.arange(12).reshape(3, 4))

        # when
        node = softmax.Softmax(x)
        node.compute()

        # then
        assert node.output.sum(axis=1).sum() == 3
        assert (node.output == [[0.64, 0.24, 0.09, 0.03], [0.64, 0.24, 0.09, 0.03], [0.64, 0.24, 0.09, 0.03]]).all()

    def test_backpass_without_upstream_gradient(self):
        # given
        x = placeholders.Variable(np.arange(6).reshape(2, 3))

        # when
        node = softmax.Softmax(x)
        node.compute()
        placeholders.GradPlaceholder(node).backpass(np.ones((2, 3)))
        node.backpass()

        # then
        assert (abs(node.gradients[x]) < 1e-8).all()

    def test_backpass_with_upstream_gradient(self):
        # given
        x_ = [[0, 2, 2],
              [3, 4, 5]]
        x = placeholders.Variable(x_)
        upstream_grad = [[0, 1, 2],
                         [3, 4, 5]]

        # when
        node = softmax.Softmax(x)
        node.compute()
        placeholders.GradPlaceholder(node).backpass(np.array(upstream_grad))
        node.backpass()

        # then
        assert (abs(node.gradients[x] - [[-0.0846, -0.1927, 0.2773], [-0.1422, -0.1392, 0.2814]]) < 1e-8).all()

