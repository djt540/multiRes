from nodes.Model import Node
import torch


class NodeArray(Node):
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.prev_state = [0] * len(self.nodes)
        if hasattr(self.nodes[0], 'num_nodes'):
            self.num_nodes = len(nodes) * nodes[0].num_nodes
        else:
            self.num_nodes = len(nodes)

    def forward(self, signal) -> torch.Tensor:
        for i in range(len(self.nodes)):
            self.prev_state[i] = self.nodes[i].forward(signal[i*100:(i+1)*100])
        return torch.flatten(torch.cat(self.prev_state))
