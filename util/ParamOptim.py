import numpy as np
from random import random, uniform
from nodes.Model import *
from dataclasses import dataclass

class ParamOpt:
    def __init__(self, model: Model, signal):
        self._signal = None

        self.model = model
        self.num_nodes = self.model.last_node.num_nodes

        self._narma = None
        self.y_train, self.y_valid, self.y_test = None, None, None
        self.signal = signal

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        self._signal = signal
        self._narma = self.model.NARMAGen(self.signal)
        _, self.y_train, self.y_valid, self.y_test = torch.split(self._narma, [250, 3750, 500, 500])

    def run(self):
        best_alpha = 0
        best_eta = 0
        best_err = 1
        best_gamma_all = 0
        for alpha in torch.arange(0.05, 0.3, 0.01):
            for eta in torch.arange(0.5, 1, 0.05):
                self.model.last_node.alpha = alpha
                self.model.last_node.eta = eta

                states = self.model.run(self.signal)  # need to pass alpha and eta to reservoir
                _, x_train, x_valid, x_test = torch.split(states, [250, 3750, 500, 500])

                gammas = torch.logspace(-9, -3, 7)
                best_gamma = 0
                local_best_error = 1

                for gamma in gammas:
                    self.model.gamma = gamma
                    error = self.model.error_test(x_train, self.y_train, x_valid, self.y_valid)
                    if error < local_best_error:
                        best_gamma = gamma
                        self.model.gamma = gamma
                        local_best_error = error

                error = self.model.error_test(x_train, self.y_train, x_test, self.y_test)

                if error < best_err:
                    best_alpha = alpha
                    best_eta = eta
                    best_err = error
                    best_gamma_all = best_gamma

        self.model.last_node.alpha = best_alpha
        self.model.last_node.eta = best_eta
        self.model.gamma = best_gamma_all

        return best_err

    @dataclass
    class Param:
        instance: object
        name: str
        min: float = 0.6
        max: float = 0.9
        step: float = 0.05
        cur_val: float = uniform(min, max)
        best_val: float = cur_val

    def anneal(self, params_list: list[Param], iterations=100, initial_temp=25):
        self.params_step(params_list)
        _, x_train, x_valid, x_test = self.split_results(self.signal)
        best_error = self.model.error_test(x_train, self.y_train, x_valid, self.y_valid)

        for i in range(iterations):
            self.params_step(params_list)
            _, x_train, x_valid, x_test = self.split_results(self.signal)

            error = self.model.error_test(x_train, self.y_train, x_valid, self.y_valid)
            error_diff = best_error - error

            acceptable = np.exp(error_diff / (initial_temp - i))

            if error_diff > 0 or random() > acceptable:
                best_error = error
                print(error)
                for param in params_list:
                    param.best_val = param.cur_val

        for param in params_list:
            param.cur_val = param.best_val

        # return Params and Error
        return best_error

    @staticmethod
    def params_step(params_list: list[Param]):
        for param in params_list:
            param.cur_val = param.best_val + uniform(-0.5, 0.5) * param.step
            setattr(param.instance, param.name, param.cur_val)
            # print(f'leak:{param.instance.leak}, in_scale:{param.instance.in_scale}')

    def split_results(self, signal, splits=None):
        if splits is None:
            splits = [250, 3750, 500, 500]
        if sum(splits) == len(signal):
            state = self.model.run(signal)
            return torch.split(state, splits)
        else:
            raise Exception("Splits do not add to same length as signal")
