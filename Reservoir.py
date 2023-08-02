from Model import *


class Reservoir(Node):
    def __init__(self, num_nodes, alpha=0.24, eta=0.67):
        self.num_nodes = num_nodes
        self.name = 'Res'

        self.alpha, self.eta = alpha, eta
        self.prev_state = 0

        self.w_res = 2 * torch.rand((num_nodes, num_nodes)) - 1
        eig_vals, eig_vecs = torch.linalg.eigh(self.w_res)
        self.w_res = self.w_res / (abs(eig_vals[0]))

    def leaky_integrator(self, prev_state, in_val):
        return (1 - self.alpha) * prev_state + self.alpha * in_val

    def forward(self, signal):
        self.prev_state = self.leaky_integrator(self.prev_state, (self.eta * signal))
        return self.prev_state

