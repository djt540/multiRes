from Model import *


class DelayLine(Node):
    def __init__(self, tau: int = 3, fb_str: float = 0.5, eta: float = 1):
        self._tau = tau
        self._fb_str = fb_str
        self.eta = eta
        self.mask = 2 * torch.rand(tau) - 1
        self._wrapped = None
        self.name = 'DelayLine'

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

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if self.wrapped is not None:
            out_list = [self.fb_str * (self.wrapped.forward(signal * self.eta * self.mask[theta]))
                        for theta in range(self.tau)]
            return torch.sum(torch.stack(out_list))
        else:
            raise Exception("Delay Line has nothing to wrap")
