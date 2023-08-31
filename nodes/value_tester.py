from nodes.Model import *
from nodes.DelayLine import DelayLine
from nodes.Rotor import Rotor
from nodes.Reservoir import Reservoir, InputMask

# this file is a bit of a mess


class BlankLine(Node):
    def __init__(self, num_nodes, verbose=True):
        self.name = 'BlankLine'

        self.verbose = verbose
        self.num_nodes = num_nodes
        self.prev_state = torch.zeros((1, num_nodes))

        self.wrapped = None

    def forward(self, signal):
        if self.wrapped is not None:
            output = self.wrapped.forward(signal)
            return output
        else:
            print(signal)
            return signal


# Rotor Mask Rotation Test
def test_rotor():
    sig = torch.flatten(torch.cat((torch.ones(100), torch.zeros(100), torch.ones(100),torch.ones(100), torch.ones(100))))
    in_mask = InputMask(500)
    in_mask.w_in = sig
    mod = Model((in_mask, Rotor(5, 100), BlankLine(500)))
    mod.run(sig)


if __name__ == '__main__':
    test_rotor()


