from nodes.Model import Node
import torch


class NodeArray(Node):
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.prev_state = [0] * len(self.nodes)
        self.num_nodes = len(nodes)
        self.obj_nodes = self.num_nodes
        if hasattr(self.nodes[0], 'num_nodes'):
            self.obj_nodes = nodes[0].num_nodes
            self.num_nodes = self.num_nodes * self.obj_nodes

    def forward(self, signal) -> torch.Tensor:
        for i in range(len(self.nodes)):
            self.prev_state[i] = self.nodes[i].forward(signal[i*self.obj_nodes:(i+1)*self.obj_nodes])
        return torch.flatten(torch.cat(self.prev_state))
