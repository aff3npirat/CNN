import numpy as np
import unittest

from src.nodes import placeholders
from src.nodes.operations import linear


class TestLinear(unittest.TestCase):

    def test_compute(self):
        # given
        a = placeholders.Variable(np.arange(12).reshape(3, 4))
        b = placeholders.Variable(np.arange(12).reshape(4, 3))
        c = placeholders.Variable(np.arange(3))

        # when
        node = linear.Linear(a, b, c)
        node.compute()

        # then
        assert (node.output == [[42, 49, 56], [114, 137, 160], [186, 225, 264]]).all()

    def test_backpass_without_upstream_gradient(self):
        # given
        x = placeholders.Variable(np.arange(50).reshape(5, 10))
        w = placeholders.Variable(np.arange(30).reshape(10, 3)+30)
        b = placeholders.Variable(np.zeros(3))

        # then
        node = linear.Linear(x, w, b)
        placeholders.GradPlaceholder(node).backpass(np.ones((5, 3), dtype=np.int))
        node.backpass()

        # when
        assert node.gradients[x].shape == (5, 10)
        assert node.gradients[w].shape == (10, 3)
        assert node.gradients[b].shape == (3,)
        assert (node.gradients[x][0, :] == w.output.sum(axis=1)).all()
        assert (node.gradients[w][:, 0] == x.output.sum(axis=0)/5).all()
        assert (node.gradients[b] == [1.0, 1.0, 1.0]).all()

    def test_backpass_with_upstream_gradient(self):
        # given
        x = placeholders.Variable(np.arange(6).reshape(2, 3))
        w = placeholders.Variable(np.arange(12).reshape(3, 4))
        b = placeholders.Variable(np.arange(4))
        upstream_grad = [[7, 6, 5, 4],
                         [3, 2, 1, -1]]

        # when
        node = linear.Linear(x, w, b)
        placeholders.GradPlaceholder(node).backpass(upstream_grad)
        node.backpass()

        # then
        assert (node.gradients[x] == [[28, 116, 204], [1, 21, 41]]).all()
        assert (node.gradients[w] == [[4.5, 3, 1.5, -1.5], [9.5, 7, 4.5, 0], [14.5, 11, 7.5, 1.5]]).all()
        assert (node.gradients[b] == [5, 4, 3, 1.5]).all()

