import multiprocessing
from datetime import date
import numpy as np
from tqdm import tqdm
import pandas as pd
from nodes.Rotor import Rotor
from nodes.DelayLine import DelayLine
from nodes.Reservoir import Reservoir
from nodes.Model import *
from util.ParamOptim import ParamOpt
from scipy import optimize
from nodes.value_tester import InputMask
from nodes.NodeArray import NodeArray


def error_average(model_desc, num_tests):
    errors = []
    template_mod = Model(model_desc)

    for x in range(num_tests):
        sig = (torch.rand(5000) / 2)
        mod = Model(model_desc)
        po = ParamOpt(mod, sig)
        errors.append(po.run())

    # with open('test-results.txt', 'a') as f:
    #     for error in errors:
    #         f.write(f'{template_mod.node_list[1].fb_str}, {template_mod.node_list[1].tau}, {error}\n')

    avrg_error = sum(errors) / num_tests
    with open('test-results.csv', 'a') as f:
        f.write(f'{template_mod.node_list[1].fb_str}, {template_mod.node_list[1].tau}, {avrg_error}\n')


def fb_tau_tester(model_desc):
    sig = torch.rand(5000) / 2

    test_size = 5
    error_mat = np.ndarray((test_size, test_size))

    mod = Model(model_desc)
    res = mod.last_node

    opt_params = [ParamOpt.Param(instance=res, name='alpha'),
                  ParamOpt.Param(instance=res, name='eta')]

    po = ParamOpt(mod, sig)

    narma = mod.NARMAGen(sig)
    _, y_train, y_valid, y_test = torch.split(narma, [250, 3750, 500, 500])

    # sweep tau and fb
    for tm in range(test_size):
        # CHANGE THIS IF THE NODE CHANGES LOCATION IN THE MODEL
        res.eta = 0.1 + (tm / test_size)
        for fb in range(test_size):
            res.reset_states()
            res.fb_str = 0.1 + (fb / test_size)
            po.anneal(opt_params)
            states = mod.run(sig)
            _, x_train, x_valid, x_test = torch.split(states, [250, 3750, 500, 500])

            w_out = mod.ridge_regression(x_train, y_train)
            pred = x_test @ w_out
            error_mat[tm][fb] = mod.NRMSE(pred, y_test)
    try:
        np.savetxt('test-results.csv', error_mat, delimiter=",", fmt='%f')
    except FileNotFoundError:
        print(error_mat)


def _tester(model_desc):
    sig = (torch.rand(5000) / 2)

    errors = []
    for i in range(1):
        mod = Model(model_desc)
        po = ParamOpt(mod, sig)

        res = mod.last_node.nodes

        narma = mod.NARMAGen(sig)
        wash, y_train, y_valid, y_test = torch.split(narma, [250, 3750, 500, 500], dim=0)

        opt_params = [ParamOpt.Param(instance=res[0], name='alpha'),
                      ParamOpt.Param(instance=res[0], name='eta'),
                      ParamOpt.Param(instance=res[1], name='alpha'),
                      ParamOpt.Param(instance=res[1], name='eta'),
                      ParamOpt.Param(instance=res[2], name='alpha'),
                      ParamOpt.Param(instance=res[2], name='eta'),
                      ParamOpt.Param(instance=res[3], name='alpha'),
                      ParamOpt.Param(instance=res[3], name='eta'),
                      ParamOpt.Param(instance=res[4], name='alpha'),
                      ParamOpt.Param(instance=res[4], name='eta'),
                      ]

        po.anneal(opt_params)

        states = mod.run(sig)
        wash, x_train, x_valid, x_test = torch.split(states, [250, 3750, 500, 500], dim=0)
        w_out = mod.ridge_regression(x_train, y_train)
        pred = x_valid @ w_out
        print(mod.NRMSE(pred, y_valid))
        mod.simple_plot(pred, y_valid)
        errors.append(torch.sum((pred - y_test) ** 2) / len(y_test))


if __name__ == "__main__":
    nnodes = 100
    # fb_tau_tester((DelayLine(tau=15), Reservoir(nnodes)))
    # _tester((Rotor(nnodes), Reservoir(nnodes)))
    res = [Reservoir(nnodes) for i in range(5)]
    # _tester((BlankLine(nnodes, verbose=False), res[1]))
    # res[1].reset_states()
    _tester((InputMask(500), DelayLine(tau=20, fb_str=0.4, eta=0.2), Rotor(5, 100), NodeArray(res)))
    # res[1].reset_states()
    # _tester((DelayLine(tau=20, fb_str=0.4, eta=0.2), res[1]))
