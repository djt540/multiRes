from nodes.Model import Node
import numpy as np


class NodeArray(Node):
    """Array of Node objects

    Notes
    -----
    If the objects in the array are reservoirs then the NodeArrays num_nodes equals number of nodes
    in the reservoir x the number of objects.

    If note the number of nodes equals the number of objects in the array.

    Parameters
    ----------
    nodes : list[Node]
        List of the nodes to for the incoming signal to be passed to. (This is effectively a wrapped list.)

    Attributes
    ----------
    obj_nodes : int
        Number of nodes each object contains, currently can only support objects with equal number of nodes.
    num_nodes: int
        Number of nodes total in the forward parts of the model, see notes for more information on conditional
        values of num_nodes.

    """
    def __init__(self, nodes: list[Node]):
        self.name = 'NodeArray'
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.obj_nodes = self.num_nodes
        if hasattr(self.nodes[0], 'num_nodes'):
            self.obj_nodes = nodes[0].num_nodes
            self.num_nodes = self.num_nodes * self.obj_nodes

    def forward(self, signal) -> np.ndarray:
        """Forward function for the node arrays, passes even slices of the input signal into the
        nodes in the nodes list, currently this is equally sized slices.

        Parameters
        ----------
        signal : np.ndarray
            Input signal for the NodeArray node
        Returns
        -------
        np.ndarray
            Output states of all the objects concatenated together.
        """
        # I am sure this could be made parallelised using the multiprocessing module or similar,
        # however I am unsure I could do that easily.
        output = [self.nodes[i].forward(signal[i*self.obj_nodes:(i+1)*self.obj_nodes]) for i in range(len(self.nodes))]
        return np.concatenate(output).ravel()
