import numpy as np
import unittest

from src.nodes import placeholders
from src.nodes.operations import mul


class TestMatMul(unittest.TestCase):

    def test_backpass_with_1D_equal_shape(self):
        # given
        a = placeholders.Variable(np.arange(10))
        b = placeholders.Variable(np.arange(10)+10)

        # when
        node = mul.MatMul(a, b)
        node.compute()
        placeholders.GradPlaceholder(node).backpass(np.ones_like(node.output))
        node.backpass()

        # then
        assert (node.gradients[a] == np.arange(10)+10).all()
        assert (node.gradients[b] == np.arange(10)).all()

    def test_backpass_with_2D_equal_shape(self):
        # given
        a = placeholders.Variable(np.arange(50).reshape(5, 10))
        b = placeholders.Variable(np.arange(50).reshape(10, 5)+50)

        # when
        node = mul.MatMul(a, b)
        placeholders.GradPlaceholder(node).backpass(np.ones((5, 5), dtype=np.int))
        node.backpass()

        # then
        assert node.gradients[a].shape == (5, 10)
        assert node.gradients[b].shape == (10, 5)
        assert (node.gradients[a][0, :] == (np.arange(50).reshape(10, 5)+50).sum(axis=1)).all()
        assert (node.gradients[b][:, 0] == np.arange(50).reshape(5, 10).sum(axis=0)).all()

    def test_backpass_with_2D_unequal_shape(self):
        # given
        a = placeholders.Variable(np.arange(50).reshape(5, 10))
        b = placeholders.Variable(np.arange(30).reshape(10, 3) + 30)

        # when
        node = mul.MatMul(a, b)
        placeholders.GradPlaceholder(node).backpass(np.ones((5, 3), dtype=np.int))
        node.backpass()

        # then
        assert node.gradients[a].shape == (5, 10)
        assert node.gradients[b].shape == (10, 3)
        assert (node.gradients[a][0, :] == (np.arange(30).reshape(10, 3) + 30).sum(axis=1)).all()
        assert (node.gradients[b][:, 0] == np.arange(50).reshape(5, 10).sum(axis=0)).all()

    def test_backpass_with_upstream_gradient(self):
        # given
        a = placeholders.Variable(np.arange(6).reshape(2, 3))
        b = placeholders.Variable(np.arange(12).reshape(3, 4))
        upstream_grad = [[7, 6, 5, 4],
                         [3, 2, 1, -1]]

        # when
        node = mul.MatMul(a, b)
        placeholders.GradPlaceholder(node).backpass(np.array(upstream_grad))
        node.backpass()

        # then
        assert (node.gradients[a] == [[28, 116, 204], [1, 21, 41]]).all()
