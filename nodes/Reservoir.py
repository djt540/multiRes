from nodes.Model import *


class Reservoir(Node):
    def __init__(self, num_nodes, alpha=0.24, eta=0.67, rho=0.95):
        self.gamma = 0.3
        self.num_nodes = num_nodes
        self.name = 'Res'

        self.alpha, self.eta, self.rho = alpha, eta, rho
        self.prev_state = torch.zeros((1, num_nodes))

        self.w_in = torch.rand((1, num_nodes))

        self.w_res = 2 * torch.rand((num_nodes, num_nodes)) - 1
        eig_vals, eig_vecs = torch.linalg.eigh(self.w_res)
        self.w_res = self.w_res / (abs(eig_vals[0]))

    # def leaky_integrator(self, prev_state, in_val):
    #     return (1 - self.alpha) * prev_state + self.alpha * in_val

    # def forward(self, signal):
    #     self.prev_state = self.leaky_integrator(self.prev_state, (self.eta * signal))
    #     return self.prev_state

    def forward(self, signal):
        self.prev_state = (1 - self.alpha) * self.prev_state + self.alpha * torch.tanh(
            self.rho * (self.prev_state @ self.w_res) + self.gamma * (signal * self.w_in))
        return self.prev_state
