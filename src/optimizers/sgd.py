from src.base import Optimizer


class SGD(Optimizer):
    """
    Stochastic gradient descend optimizer.

    Updates weights with w_new = w_old - learning_rate*dw_old.
    """

    def __init__(self, learning_rate):
        """
        Args:
            learning_rate (float)
        """
        super().__init__(learning_rate)

    def update(self, trainables):
        """
        Updates each given Variable using the respective gradient.

        Args:
            trainables (list/tuple[Variables]): Variables to be updated.
        """
        for node in trainables:
            node.output += -self.learning_rate * node.gradients[node]
