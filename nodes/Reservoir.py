import torch

from nodes.Model import *


class InputMask(Node):
    def __init__(self, num_nodes):
        self.w_in = torch.ones(num_nodes)
        self.name = 'Input Mask'

        self.wrapped = None

    def forward(self, signal) -> torch.Tensor:
        return self.wrapped.forward(signal * self.w_in)


class Reservoir(Node):
    def __init__(self, num_nodes, connectivity=0.1, leak: float = 0.85, in_scale: float = 0.25, spec_r: float = 0.85):
        self.name = 'Res'
        # here for deep ESN
        self.wrapped = None

        self.num_nodes = num_nodes
        self._leak, self._in_scale, self.spec_r = leak, in_scale, spec_r

        self.prev_state = torch.zeros(num_nodes)

        self.w_res = self._internal_weights_calc(connectivity)

    def _internal_weights_calc(self, connectivity):
        weights = torch.rand((self.num_nodes, self.num_nodes))
        weights[weights < connectivity] = 0
        vals, vecs = torch.linalg.eig(weights)
        max_eig = torch.max(torch.abs(vals[0]))
        weights /= torch.abs(max_eig) / self.spec_r
        return weights

    def reset_states(self):
        self.prev_state = torch.zeros(self.num_nodes)

    def forward(self, signal) -> torch.Tensor:
        leak_in = (1 - self._leak) * self.prev_state
        sig_in = self._leak * self.prev_state @ self.w_res + self._in_scale * signal
        self.prev_state = leak_in + torch.tanh(sig_in)
        return self.prev_state

    # state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T)

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
