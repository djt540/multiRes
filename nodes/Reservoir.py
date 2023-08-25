from nodes.Model import *


class InputMask(Node):
    def __init__(self, num_nodes):
        self.w_in = torch.rand(num_nodes)
        self.name = 'Input Mask'

        self.wrapped = None

    def forward(self, signal) -> torch.Tensor:
        return self.wrapped.forward(signal * self.w_in)


class Reservoir(Node):
    def __init__(self, num_nodes, sparsity=0.8, leak: float = 1, in_scale: float = 0.7, spec_r: float = 0.99):
        self.name = 'Res'

        self.num_nodes = num_nodes
        self._leak, self._in_scale, self.spec_r = leak, in_scale, spec_r

        self.prev_state = torch.zeros(num_nodes)

        self.w_res = torch.rand((num_nodes, num_nodes))

        self.w_res[self.w_res < sparsity] = 0
        vals, vecs = torch.linalg.eig(self.w_res)
        self.w_res = self.w_res / torch.abs(vals[0])

    def reset_states(self):
        self.prev_state = torch.zeros((1, self.num_nodes))

    def forward(self, signal) -> torch.Tensor:
        self.prev_state = (1 - self.leak) * self.prev_state + self.leak * torch.tanh(
            self.spec_r * self.prev_state @ self.w_res + self.in_scale * signal)
        return self.prev_state

    # def leaky_integrator(self, in_val):
    #     return (1 - self.alpha) * self.prev_state + self.alpha * in_val
    #
    # def forward(self, signal) -> torch.Tensor:
    #     self.prev_state = self.leaky_integrator(self.eta * signal)
    #     return self.prev_state

    @property
    def leak(self):
        return self._leak

    @leak.setter
    def leak(self, leak):
        self._leak = leak

    @property
    def in_scale(self):
        return self._in_scale

    @in_scale.setter
    def in_scale(self, in_scale):
        self._in_scale = in_scale

    def str(self):
        return f'{self._leak=} {self._in_scale=}'
