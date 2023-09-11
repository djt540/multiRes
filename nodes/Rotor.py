from nodes.Model import *


class Rotor(Node):
    """Rotates inputs signal and output states
    Based on: Rotating neurons for all-analog implementation of cyclic reservoir computing [RNR]_

    Notes
    -----
        *This Node is aimed to work with the NodeArray node, where Rotor rotates the
        incoming signals and outgoing states the NodeArray passes these signals onto
        multiple Nodes, splitting the signal evenly between the nodes.*

    Citations
    ---------
    .. [RNR] Liang, X., Zhong, Y., Tang, J. et al. Rotating neurons for all-analog
        implementation of cyclic reservoir computing. Nat Commun 13, 1549 (2022).
        https://doi.org/10.1038/s41467-022-29260-1

    Parameters
    ----------
    rot_num : int
        Number of Nodes in the system
    tot_inputs : int
        Total number of inputs in the following node array
    """
    def __init__(self, rot_num: int, tot_inputs: int):
        self.name = 'Rotor'
        self._wrapped = None

        self._rot_num = rot_num
        self._tot_inputs = tot_inputs
        self.roll_amount = int(self._tot_inputs / self._rot_num)
        self._roll_count = 0

    def forward(self, signal) -> np.ndarray:
        """Takes in a signal of size n where n is number of inputs per object x number
        of objects. So for 3 ESNs with 100 inputs each rotation will be of size 100.
        The rotation amount is the product of the rotation amount and the current timestep.

        Parameters
        ----------
        signal : np.ndarray
            Input signal for the rotor node.
        Returns
        -------
        np.ndarray
            Output States
        """
        if self._roll_count < self._rot_num:
            self._roll_count = 0
        self._roll_count += 1
        current_roll = self.roll_amount * self._roll_count

        state = self.wrapped.forward(np.roll(signal, current_roll))
        output = np.roll(state, -current_roll)
        return output

    @property
    def wrapped(self) -> Node:
        """Node that the forward function passes the signal to.
        Returns
        -------
        Node
            The node that this node currently wraps in the model
        """
        return self._wrapped

    @wrapped.setter
    def wrapped(self, node: Node):
        self._wrapped = node
