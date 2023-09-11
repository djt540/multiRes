from nodes.Model import Node
import numpy as np


class DelayLine(Node):
    """Implements virtual nodes using a simple mask and time multiplexing.
    Based on: "Information processing using a single dynamical node as complex system"
    [SDN]_

    Citations
    ---------
    .. [SDN] Appeltant, L., Soriano, M., Van der Sande, G. et al. Information processing using a single dynamical
        node as complex system. Nat Commun 2, 468 (2011). https://doi.org/10.1038/ncomms1476

    Parameters
    ----------
    tau : int
        Number of virtual nodes
    fb_str : float
        Hyperparameter Feedback strength
    eta : float
        Hyperparameter input scaling

    Attributes
    ----------
    _mask : np.ndarray
        Mask the size of the virtual nodes applied to the signal in the forward function.

        This is not the mask used in the paper, output is highly dependent on the mask,
        however this is the best mask I could find for this task.
    _wrapped : Node|None
        Node that the forward function passes the signal to.
    _v_states : np.ndarray
        Internal states of the virtual nodes, only contains one timestep before overwritten
    """

    def __init__(self, tau: int = 3, fb_str: float = 0.5, eta: float = 0.1):
        self.name = 'DelayLine'
        self._mask = 2 * (np.random.randint(0, 2, tau) - 0.5)
        self._tau, self._fb_str, self.eta = tau, fb_str, eta
        self._wrapped: Node | None = None
        self._v_states = [0] * self.tau

    def forward(self, signal: np.ndarray) -> np.ndarray:
        """
        Forward function of the delay line. Per timestep a mask is applied to the incoming signal and added to the
        product of the previous state and feedback strength. For the number of virtual nodes the modified signal is
        through a dynamical node (the wrapped nodes forward function). The final state from v_states is returned.

        Parameters
        ----------
        signal : np.ndarray
            Input signal for the delay line node.
        Raises
        ------
        Exception
            Delay Line has nothing to wrap if wrapped = None
        Returns
        -------
        np.ndarray
            Returns output states (the delay lines final node output) from each node in the chain
        """
        if self.wrapped is not None:
            for theta in range(self.tau):
                # Had to add 1 to eta and fb_str, unsure why.
                masked_value = (signal * (1 + self.eta) * self._mask[theta])
                old_state = (self._v_states[theta] * (1 + self.fb_str))
                self._v_states[theta] = self.wrapped.forward(masked_value + old_state)
            return np.array(self._v_states).sum(axis=0)
            # return self._v_states[self.tau-1] # Couldn't get this working as well
        else:
            raise Exception("Delay Line has nothing to wrap")

    @property
    def fb_str(self):
        """

        Returns
        -------
        Feedback strength for the delay line
        """
        return self._fb_str

    @fb_str.setter
    def fb_str(self, fb_str):
        self._fb_str = fb_str

    @property
    def tau(self):
        """

        Returns
        -------
        Number of virtual nodes in the delay line
        """
        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        self._mask = (10 * (np.random.randint(0, 2, tau) + 1)) - 5

    @property
    def wrapped(self) -> Node:
        """Node that the forward function passes the signal to.
        Returns
        -------
        The node that this node currently wraps in the model
        """
        return self._wrapped

    @wrapped.setter
    def wrapped(self, node: Node):
        self._wrapped = node
