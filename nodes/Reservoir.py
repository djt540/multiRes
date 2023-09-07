import numpy as np

from nodes.Model import *


class InputMask(Node):
    """Applies an input mask of size num_nodes with a uniform distribution of 1 and -1.

    Parameters
    ----------
    num_nodes : int
        The number of nodes for the mask.

    Attributes
    ----------
    w_in : np.ndarray
        The mask that is applied to the incoming signal.
    _wrapped : Node|None
        Node that the forward function passes the signal to.
    """

    def __init__(self, num_nodes, connectivity: float = 0.5):
        np.random.seed(seed=1)
        self.w_in = np.random.rand(num_nodes,)
        self.w_in[self.w_in < connectivity] = 0
        self.w_in -= 0.5
        self.name = 'Input Mask'

        self._wrapped = None

    def forward(self, signal) -> np.ndarray:
        """
        Applies the input mask to the signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal for the delay line node.

        Returns
        -------
        np.ndarray
            Returns output states including changes from future nodes
        """
        return self.wrapped.forward(signal * self.w_in)

    @property
    def wrapped(self) -> Node:
        """ Node that the forward function passes the signal to.
        Returns
        -------
        The node that this node currently wraps in the model
        """
        return self._wrapped

    @wrapped.setter
    def wrapped(self, node: Node):
        self._wrapped = node


class Reservoir(Node):
    """An implementation of an Echo State Network

    Parameters
    ----------
    num_nodes : int
        The size of the reservoir - the number of nodes in the reservoir.
        Typically larger reservoirs have higher capacities.
    connectivity : float
        The connectivity percentage of the reservoirs weight matrix, with
        each connection less than the connectivity being set to 0 in weight instantiation.
    leak : float
        The rate at which previous state feedback affects the reservoir.
    in_scale : float
        The rate at which the current feedback affects the state of the reservoir.
    spec_r : float
        The spectral radius of the weight matrix of the reservoir.

    Attributes
    ----------
    wrapped : Node|None
        Node that the forward function passes the signal to.
        Note: currently goes nowhere, however I believe allowing DeepESNs should be easy
        with this framework.
    prev_state : np.ndarray
        State of the reservoir in its perceived last timestep (i.e. the last time a
        signal was passed in). This is used for the update/forward calculation.
    w_res : np.ndarray
        Internal weights of the reservoir.
    """

    def __init__(self, num_nodes: int, connectivity: float = 0.1, leak: float = 0.75, in_scale: float = 1,
                 spec_r: float = 0.95):
        np.random.seed(seed=1)
        self.name = 'Res'
        self.wrapped = None

        self.num_nodes = num_nodes
        self._leak, self._in_scale, self.spec_r = leak, in_scale, spec_r

        self.prev_state = np.zeros(num_nodes)
        self.w_res = self._internal_weights_calc(connectivity)

    def _internal_weights_calc(self, connectivity):
        weights = np.random.rand(self.num_nodes, self.num_nodes)
        weights[weights < connectivity] = 0
        vals, vecs = np.linalg.eig(weights)
        max_eig = np.max(np.abs(vals[0]))
        weights = weights * (self.spec_r / max_eig)
        return weights

    def reset_states(self):
        self.prev_state = np.zeros(self.num_nodes)

    def forward(self, signal) -> np.ndarray:
        """Forward function of the reservoir
            Following the following update equation:
            .. math::
                x(n+1)= (1-leak) \cdot x_n + tanh(leak * x_n \times W_{res} + in \textunderscore scale * signal)
        Parameters
        ----------
        signal
            Input signal for the reservoir node

        Returns
        -------
            Output states for the reservoir
        """
        leak_in = (1 - self.leak) * self.prev_state
        sig_in = self.leak * self.prev_state @ self.w_res + self._in_scale * signal
        self.prev_state = leak_in + np.tanh(sig_in)
        return self.prev_state

    @property
    def leak(self):
        """Leak rate for previous states
        Returns
        -------
        Value of the Leak property
        """
        return self._leak

    @leak.setter
    def leak(self, leak):
        self._leak = leak

    @property
    def in_scale(self):
        """Scaling for the input into the reservoir
        Returns
        -------
        float
            value of in_scale
        """
        return self._in_scale

    @in_scale.setter
    def in_scale(self, in_scale):
        self._in_scale = in_scale

    def __str__(self):
        return f'{self._leak=} {self._in_scale=}'
