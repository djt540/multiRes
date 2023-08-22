from nodes.Model import *
from nodes.DelayLine import DelayLine
from nodes.Rotor import Rotor
from nodes.Reservoir import Reservoir

# this file is a bit of a mess


class TestLine(Node):
    def __init__(self, num_nodes, verbose=True):
        self.name = 'TestLine'

        self.verbose = verbose
        self.num_nodes = num_nodes
        self.prev_state = torch.zeros((1, num_nodes))

        self.wrapped = None

    def forward(self, signal, fb_str=1):
        self.prev_state = self.wrapped.forward(signal)
        if self.verbose:
            print(self.prev_state)
        return self.prev_state


if __name__ == '__main__':
    # sig = torch.ones(5)
    sig = torch.rand(5000)
    model_test = Model((DelayLine(tau=120), TestLine(2, verbose=False)))

    del_line = model_test.node_list[0]
    del_line.eta = 0.5
    del_line.fb_str = 0.5
    del_line.mask = torch.rand(del_line.tau)

    model_test = Model((DelayLine(tau=5, fb_str=0.5), Reservoir(5)))
    # model_test = Model((Rotor(3), Reservoir(3)))

    out = model_test.run(sig)
    print(out)
