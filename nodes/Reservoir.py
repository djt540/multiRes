from nodes.Model import *


class Reservoir(Node):
    def __init__(self, num_nodes, sparsity=0.8, alpha: float = 0.9, eta: float = 0.7, rho: float = 0.9):
        self.name = 'Res'

        self.num_nodes = num_nodes
        self._alpha, self._eta, self.rho = alpha, eta, rho

        self.prev_state = torch.zeros(num_nodes)

        self.w_in = torch.rand((1, num_nodes))
        self.w_res = torch.rand((num_nodes, num_nodes))

        self.w_res[self.w_res < sparsity] = 0
        vals, vecs = torch.linalg.eig(self.w_res)
        self.w_res = self.w_res / torch.abs(vals[0])

    def reset_states(self):
        self.prev_state = torch.zeros((1, self.num_nodes))

    def forward(self, signal) -> torch.Tensor:
        self.prev_state = (1 - self.alpha) * self.prev_state + self.alpha * torch.tanh(
            self.rho * self.prev_state @ self.w_res + self.eta * self.w_in * signal)
        return self.prev_state

    # def leaky_integrator(self, in_val):
    #     return (1 - self.alpha) * self.prev_state + self.alpha * in_val
    #
    # def forward(self, signal) -> torch.Tensor:
    #     self.prev_state = self.leaky_integrator(self.eta * signal)
    #     return self.prev_state

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
