import numpy as np
import unittest

from src.nodes import placeholders
from src.optimizers import adam


def loss_func(x):
    return x ** 2 - 2 * x + 1


def grad_func(x):
    return 2*x - 2


class TestAdam(unittest.TestCase):

    def test_update(self):
        # given
        x = placeholders.Variable(np.array(0.0))
        optimizer = adam.Adam(learning_rate=0.1)

        # when
        t = 0
        while True:
            t += 1
            x.gradients[x] = grad_func(x.output)
            optimizer.update([x])
            if abs(loss_func(x.output)) < 1e-3:
                print(f"converged after {t} iterations")
                break
            else:
                print(f"iteration {t}: x={x.output}")

        # then
        assert t == optimizer._iterations

