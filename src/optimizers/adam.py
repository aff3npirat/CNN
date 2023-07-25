import numpy as np

from src.base import Optimizer


class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-7, t_init=0):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_hat = eps
        self._iterations = t_init
        self._m = {}
        self._v = {}

    def update(self, trainables):
        self._iterations += 1
        for node in trainables:
            if node not in self._m.keys():
                self._m[node] = np.zeros_like(node.output)
                self._v[node] = np.zeros_like(node.output)

            grad = node.gradients[node]
            m = self.beta1 * self._m[node] + (1 - self.beta1) * grad
            v = self.beta2 * self._v[node] + (1 - self.beta2) * np.power(grad, 2)

            beta_1_power = np.power(self.beta1, self._iterations)
            beta_2_power = np.power(self.beta2, self._iterations)
            alpha_t = self.learning_rate * (np.sqrt(1 - beta_2_power)/(1 - beta_1_power))
            node.output += -alpha_t * (m / (np.sqrt(v) + self.eps_hat))
