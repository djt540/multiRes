from nodes.Model import *
import numpy as np


class DelayLine(Node):
    def __init__(self, tau: int = 3, fb_str: float = 0.5, eta: float = 1):
        self.name = 'DelayLine'

        self._tau, self._fb_str, self.eta = tau, fb_str, eta
        self.mask = torch.rand(tau)

        self._wrapped = None

    def forward(self, signal: torch.Tensor, fb_str: float = 1) -> torch.Tensor:
        if self.wrapped is not None:
            out_list = [(self.wrapped.forward(signal * self.eta * self.mask[theta], self._fb_str))
                        for theta in range(self.tau)]
            return torch.sum(torch.stack(out_list), dim=0)
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
        self.mask = torch.rand(tau)

    @property
    def wrapped(self):
        return self._wrapped

    @wrapped.setter
    def wrapped(self, node: Node):
        self._wrapped = node
