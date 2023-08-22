from nodes.Model import *


class Reservoir(Node):
    def __init__(self, num_nodes, alpha: float = 0.24, eta: float = 0.67, rho: float = 0.95, up_b: float = 1,
                 low_b: float = 1):
        self.name = 'Res'

        self.num_nodes = num_nodes
        self._alpha, self._eta, self.rho = alpha, eta, rho
        self._up_b, self._low_b = up_b, low_b

        self.prev_state = torch.zeros((1, num_nodes))

        self.w_in = torch.rand((1, num_nodes))
        self.w_res = (self._low_b - self._up_b) * torch.rand((num_nodes, num_nodes)) + self._up_b
        eig_vals, eig_vecs = torch.linalg.eigh(self.w_res)
        self.w_res = self.w_res / (abs(eig_vals[0]))

    def reset_states(self):
        self.prev_state = torch.zeros((1, self.num_nodes))

    def forward(self, signal, fb_str: float = 1):
        self.prev_state = (1 - self.alpha) * self.prev_state + self.alpha * torch.tanh(
            self.rho * (self.prev_state * fb_str @ self.w_res) + self.eta * (signal * self.w_in))
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
