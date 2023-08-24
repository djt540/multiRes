from nodes.Model import *


class Rotor(Node):
    def __init__(self, rot_num, num_nodes: int):
        self.name = 'Rotor'
        self.rot_num = rot_num
        self.num_nodes = num_nodes
        self.w_in = torch.rand((1, num_nodes * rot_num))
        self.mask = torch.randint(-1, 1, (rot_num, 1))
        self.roll_count = 0

        self._wrapped = None

    def forward(self, signal) -> torch.Tensor:
        state = self.wrapped.forward(torch.roll(signal, self.roll_count * self.num_nodes))
        output = torch.roll(state, -self.roll_count * self.num_nodes)
        self.roll_count += 1
        return output

    # def forward(self, signal):
    #     if self.wrapped is not None:
    #         if self.roll_count >= self.num_nodes:
    #             self.roll_count = 0
    #         print(self.num_nodes)
    #         state = self.wrapped.forward(signal * torch.roll(self.mask, self.roll_count * self.num_nodes))
    #         self.roll_count += 1
    #         return torch.roll(state, -self.roll_count)
    #     else:
    #         raise Exception("Rotor has nothing to wrap")

    @property
    def wrapped(self):
        return self._wrapped

    @wrapped.setter
    def wrapped(self, node: Node):
        self._wrapped = node
