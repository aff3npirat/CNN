import unittest

from src.utils import core


class TestCategoricalToOneHot(unittest.TestCase):

    def test_categorical2_one_hot(self):
        # given
        categories = [0, 1, 1, 4, 5, 0]

        # when
        one_hot = core.categorical2one_hot(categories, 10)

        # then
        expected_one_hot = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert (expected_one_hot == one_hot).all()
