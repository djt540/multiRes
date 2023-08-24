from nodes.Model import Node
from nodes.Reservoir import Reservoir
import torch


class ResArray(Node):
    def __init__(self, reservoirs: list[Reservoir]):
        self.reservoirs = reservoirs
        self.prev_state = [0] * len(self.reservoirs)
        self.num_nodes = len(reservoirs) * reservoirs[0].num_nodes

    def forward(self, signal) -> torch.Tensor:
        for i in range(len(self.reservoirs)):
            self.prev_state[i] = self.reservoirs[i].forward(signal[i])
        return torch.flatten(torch.cat(self.prev_state))
