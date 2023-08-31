from nodes.Model import Node
import torch


class DelayLine(Node):
    """Implementation of a delay line node
    Based on: "Information processing using a single dynamical node as complex system"
    (https://doi.org/10.1038/ncomms1476)

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
    _mask : torch.Tensor
        Mask the size of the virtual nodes applied to the signal in the forward function
    _wrapped : None|Node
        Node that the forward function passes the signal to.
    _v_states : torch.Tensor
        Internal states of the virtual nodes, only contains one timestep before overwritten
    """

    def __init__(self, tau: int = 3, fb_str: float = 0.5, eta: float = 1):
        self.name = 'DelayLine'
        self._mask = torch.rand(tau)
        self._tau, self._fb_str, self.eta = tau, fb_str, eta
        self._wrapped: Node | None = None
        self._v_states = [0] * self.tau

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the delay line. Per timestep a mask is applied to the incoming signal and added to the
        product of the previous state and feedback strength. For the number of virtual nodes the modified signal is
        through a dynamical node (the wrapped nodes forward function). The final state from v_states is returned.

        Parameters
        ----------
        signal : torch.Tensor
            input signal for the delay line node.
        Raises
        ------
        Exception
            Delay Line has nothing to wrap if wrapped = None
        Returns
        -------
        torch.Tensor
            Returns modified signal from each node in the chains forward function (the delay lines final node output)
        """
        if self.wrapped is not None:
            for theta in range(self.tau):
                masked_value = (signal * self.eta * self._mask[theta])
                old_state = (self._v_states[theta] * self.fb_str)
                self._v_states[theta] = self.wrapped.forward(masked_value + old_state)
            return self._v_states[self.tau - 1]
        else:
            raise Exception("Delay Line has nothing to wrap")

    @property
    def fb_str(self):
        return self._fb_str

    @fb_str.setter
    def fb_str(self, fb_str):
        self._fb_str = fb_str

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        self._mask = torch.rand(tau)

    @property
    def wrapped(self):
        return self._wrapped

    @wrapped.setter
    def wrapped(self, node: Node):
        self._wrapped = node
