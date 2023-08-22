from nodes.Model import *


class Rotor(Node):
    def __init__(self, num_nodes: int):
        self.name = 'Rotor'

        self.mask = 2 * torch.rand(num_nodes) - 1
        self.roll_count = 0

        self._wrapped = None

    def forward(self, signal, fb_str: float = 1):
        if self.wrapped is not None:
            state = self.wrapped.forward(signal * torch.roll(self.mask, self.roll_count))
            self.roll_count += 1
            return torch.roll(state, -self.roll_count)
        else:
            raise Exception("Rotor has nothing to wrap")

    @property
    def wrapped(self):
        return self._wrapped

    @wrapped.setter
    def wrapped(self, node: Node):
        self._wrapped = node