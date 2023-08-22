from nodes.Model import *


class Reservoir(Node):
    def __init__(self, num_nodes, alpha: float = 0.1, eta: float = 0.1):
        self.name = 'Res'

        self.num_nodes = num_nodes
        self.node_mask = 2 * torch.rand(num_nodes) - 1
        self._alpha, self._eta = alpha, eta

        self.prev_state = torch.zeros((1, num_nodes))


    def reset_states(self):
        self.prev_state = torch.zeros((1, self.num_nodes))

    def leaky_integrator(self, in_val, fb_str: float):
        return (1 - self.alpha) * self.prev_state * fb_str + self.alpha * in_val

    def forward(self, signal, fb_str: float = 1):
        self.prev_state = self.leaky_integrator((self.eta * signal * self.node_mask), fb_str)
        return self.prev_state

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        self._eta = eta

    def str(self):
        return f'{self.alpha=} {self.eta=}'
