from Rotor import Rotor
from DelayLine import DelayLine
from Reservoir import Reservoir
from Model import *
from ParamOptim import ParamOpt
import multiprocessing
from datetime import date


def error_average(model_desc, num_tests):
    errors = []
    for x in range(num_tests):
        sig = (torch.rand(5000) / 2)
        mod = Model(model_desc)
        po = ParamOpt(mod)
        errors.append(po.run(sig))

    avrg_error = sum(errors) / num_tests
    with open('test-results.txt', 'a') as f:
        today = date.today()
        template_mod = Model(model_desc)
        f.write(f'Time:{today}  Model: {template_mod.__str__()} Error: {avrg_error}\n')


if __name__ == "__main__":
    time_multi = 5
    nnodes = 100
    feedback = 0.5

    s = (torch.rand(5000) / 2)

    reservoir = Reservoir(nnodes)

    # models_list = [(DelayLine(time_multi, feedback), reservoir),
    #                (Rotor(nnodes), reservoir),
    #                (DelayLine(time_multi, feedback), Rotor(nnodes), reservoir),
    #                (Rotor(nnodes), DelayLine(time_multi, feedback), reservoir)]

    models_list = [(DelayLine(time_multi, fb), reservoir) for fb in torch.arange(0.5, 0.75, 0.05)]

    print(models_list)
    processes = []
    for model in models_list:
        p = multiprocessing.Process(target=error_average, args=(model, 5))
        processes.append(p)
        p.start()
