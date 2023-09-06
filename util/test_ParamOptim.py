import pytest
from util.ParamOptim import ParamOpt
import numpy as np
from nodes.DelayLine import DelayLine
from nodes.Model import *
from nodes.Reservoir import Reservoir


def basic_po():
    sig = (np.rand(5000) / 2)
    res = Reservoir(10)
    print(res)
    mod = Model((DelayLine(tau=15), res))
    return ParamOpt(mod, sig), res


def test_params_step():
    po, res = basic_po()

    test_list = [ParamOpt.Param(instance=res, name='alpha'),
                 ParamOpt.Param(instance=res, name='eta')]

    po.params_step(test_list)

    print(res)

    assert res.alpha == test_list[0].cur_val
    assert res.eta == test_list[1].cur_val


def test_anneal():
    pass
