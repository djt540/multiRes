from Rotor import Rotor
from DelayLine import DelayLine
from Reservoir import Reservoir
from Model import *
from ParamOptim import ParamOpt
import multiprocessing
from datetime import date
import numpy as np
import pandas as pd


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
    sig = (torch.rand(5000) / 2)

    test_size = 5
    error_mat = np.ndarray((test_size, test_size))

    mod = Model(model_desc)
    res = mod.last_node

    opt_params = [
        dict(obj=res.alpha, min=0.05, max=0.3, step=0.01),
        dict(obj=res.eta, min=0.5, max=1, step=0.05),
    ]

    po = ParamOpt(mod, sig)

    narma = mod.NARMAGen(sig, 10)
    _, y_train, y_valid, y_test = torch.split(narma, [250, 3750, 500, 500])

    # sweep tau and fb
    for tm in range(test_size):
        # CHANGE THIS IF THE NODE CHANGES LOCATION IN THE MODEL
        mod.node_list[0].eta = 1 + (tm/test_size)
        for fb in range(test_size):
            mod.node_list[0].fb_str = 0.4 + (fb/test_size)

            po.anneal(opt_params)
            states = mod.run(sig)
            _, x_train, x_valid, x_test = torch.split(states, [250, 3750, 500, 500])

            w_out = mod.RidgeRegression(x_train, y_train)
            pred = x_test @ w_out
            error_mat[tm][fb] = torch.sum((pred - y_valid) ** 2) / len(y_valid)

    np.savetxt('test-results.csv', error_mat, delimiter=",", fmt='%f')


def rotor_tester(model_desc):
    sig = (torch.rand(1, 5000) / 2)

    errors = []
    for i in range(10):
        mod = Model(model_desc)
        res = mod.last_node
        po = ParamOpt(mod, sig)

        opt_params = [
            dict(obj=res.alpha, min=0.05, max=0.3, step=0.01),
            dict(obj=res.eta, min=0.5, max=1, step=0.05),
        ]

        narma = mod.NARMAGen(sig, 10)
        _, y_train, y_valid, y_test = torch.split(narma, [250, 3750, 500, 500])

        po.anneal(opt_params)

        states = mod.run(sig)
        _, x_train, x_valid, x_test = torch.split(states, [250, 3750, 500, 500])

        w_out = mod.RidgeRegression(x_train, y_train)
        pred = x_test @ w_out
        print(torch.sum((pred - y_test) ** 2) / len(y_test))
        errors.append(torch.sum((pred - y_test) ** 2) / len(y_test))

    avrg_error = sum(errors) / 10

    print(avrg_error)


if __name__ == "__main__":
    nnodes = 1
    fb_tau_tester((DelayLine(tau=15), Reservoir(nnodes)))
    # rotor_tester((Rotor(nnodes), Reservoir(nnodes)))
