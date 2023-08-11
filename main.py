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
        po = ParamOpt(mod)
        errors.append(po.run(sig))

    # with open('test-results.txt', 'a') as f:
    #     for error in errors:
    #         f.write(f'{template_mod.node_list[1].fb_str}, {template_mod.node_list[1].tau}, {error}\n')

    avrg_error = sum(errors) / num_tests
    with open('test-results.csv', 'a') as f:
        f.write(f'{template_mod.node_list[1].fb_str}, {template_mod.node_list[1].tau}, {avrg_error}\n')


def fb_tau_tester(model_desc):
    sig = (torch.rand(5000) / 2)

    test_size = 10
    error_mat = np.ndarray((test_size, test_size))

    mod = Model(model_desc)
    po = ParamOpt(mod)
    # po.run(sig)

    narma = mod.NARMAGen(sig, 10)
    _, y_train, y_valid, y_test = torch.split(narma, [250, 3750, 500, 500])

    # sweep tau and fb
    for tm in tqdm(range(test_size)):
        # CHANGE THIS IF THE NODE CHANGES LOCATION IN THE MODEL
        mod.node_list[0].eta = 0.1 + (tm/test_size)
        for fb in range(test_size):
            mod.node_list[0].fb_str = 0.1 + (fb/test_size)

            po.run(sig)
            states = mod.run(sig)
            _, x_train, x_valid, x_test = torch.split(states, [250, 3750, 500, 500])

            w_out = mod.RidgeRegression(x_train, y_train)
            pred = x_test @ w_out
            error_mat[tm][fb] = torch.sum((pred - y_valid) ** 2) / len(y_valid)

    np.savetxt('test-results.csv', error_mat, delimiter=",", fmt='%f')


if __name__ == "__main__":
    nnodes = 100
    s = (torch.rand(5000) / 2)

    reservoir = Reservoir(nnodes)

    # models_list = [(DelayLine(time_multi, feedback), reservoir),
    #                (Rotor(nnodes), reservoir),
    #                (DelayLine(time_multi, feedback), Rotor(nnodes), reservoir),
    #                (Rotor(nnodes), DelayLine(time_multi, feedback), reservoir)]

    # models_list = [[(Rotor(nnodes), DelayLine(tm, fb), reservoir) for tm in torch.arange(1, 20, 1)]
    #                for fb in torch.arange(0.5, 1.1, 0.02)]
    #
    # models_list = [val for sublist in models_list for val in sublist]

    # models_list = [(Rotor(nnodes), DelayLine(), reservoir),
    #                (Rotor(nnodes), DelayLine(), reservoir),
    #                (Rotor(nnodes), DelayLine(), reservoir),
    #                (Rotor(nnodes), DelayLine(), reservoir)]
    #
    # print(models_list)
    #
    # processes = []
    # for model in models_list:
    #     p = multiprocessing.Process(target=fb_tau_tester, args=(model,))
    #     processes.append(p)
    #     p.start()

    fb_tau_tester((DelayLine(tau=3), reservoir))

