from nodes.Model import *
import numpy as np


class DelayLine(Node):
    def __init__(self, tau: int = 3, fb_str: float = 0.5, eta: float = 1):
        self.name = 'DelayLine'
        self.mask = torch.randint(-1, 1, (tau, 1))
        self._tau, self._fb_str, self.eta = tau, fb_str, eta
        self._wrapped = None
        self.v_states = [0] * self.tau

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if self.wrapped is not None:
            for theta in range(self.tau):
                masked_value = (signal * self.eta * self.mask[theta])
                old_state = (self.v_states[theta] * self.fb_str)
                new_state = self.wrapped.forward(masked_value + old_state)
                self.v_states[theta] = new_state
            return new_state
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
        self.mask = torch.randint(-1, 1,tau)

    @property
    def wrapped(self):
        return self._wrapped

    @wrapped.setter
    def wrapped(self, node: Node):
        self._wrapped = node
