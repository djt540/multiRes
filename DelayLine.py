from Model import *


class DelayLine(Node):
    def __init__(self, tau: int, fb_str: float = 0.5, in_scale: float = 0):
        self.tau = tau
        self.fb_str = fb_str
        self.in_scale = in_scale
        self.mask = 2 * torch.rand(tau) - 1
        self._wrapped = None
        self.name = 'DelayLine'

    @property
    def wrapped(self):
        return self._wrapped

    @wrapped.setter
    def wrapped(self, node: Node):
        self._wrapped = node

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if self.wrapped is not None:
            out_list = [self.fb_str * (self.wrapped.forward(signal * self.mask[theta]) * -self.mask[theta])
                        for theta in range(self.tau)]
            return torch.sum(torch.stack(out_list))
        else:
            raise Exception("Delay Line has nothing to wrap")


def forward(self, signal: torch.Tensor) -> torch.Tensor:
    out_list = []
    for theta in range(self.tau):
        out_list.append(self.wrapped.forward(signal * self.mask[theta]) * -self.mask[theta])
    return torch.sum(torch.stack(out_list))
