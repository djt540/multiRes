import torch
import numpy as np
from random import random, randrange
from Model import *


class ParamOpt:
    def __init__(self, model: Model):
        self.signal = None
        self.model = model
        self.num_nodes = self.model.last_node.num_nodes
        self.narma = self.model.NARMAGen(self.signal, 10)
        _, self.y_train, self.y_valid, self.y_test = torch.split(self.narma, [250, 3750, 500, 500])

    def run(self, signal: torch.Tensor):
        self.signal = signal
        best_alpha = 0
        best_eta = 0
        best_err = 1
        best_gamma_all = 0
        for alpha in torch.arange(0.05, 0.3, 0.01):
            for eta in torch.arange(0.5, 1, 0.05):
                self.model.last_node.alpha = alpha
                self.model.last_node.eta = eta

                states = self.model.run(signal)  # need to pass alpha and eta to reservoir
                _, x_train, x_valid, x_test = torch.split(states, [250, 3750, 500, 500])

                gammas = torch.logspace(-9, -3, 7)
                best_gamma = 0
                local_best_error = 1

                for gamma in gammas:
                    self.model.gamma = gamma
                    error = self.error_test(x_train, x_valid)
                    if error < local_best_error:
                        best_gamma = gamma
                        self.model.gamma = gamma
                        local_best_error = error

                error = self.error_test(x_train, x_test)

                if error < best_err:
                    best_alpha = alpha
                    best_eta = eta
                    best_err = error
                    best_gamma_all = best_gamma

        self.model.last_node.alpha = best_alpha
        self.model.last_node.eta = best_eta
        self.model.gamma = best_gamma_all

        return best_err

    def anneal(self, signal, params_dict: dict, iterations=1000, initial_temp=100):

        best_params = [randrange(param[1, 2]) * param[3] for param in params_dict]
        self.params_update(params_dict, best_params)

        _, x_train, x_valid, _ = self.split_results(signal)
        best_error = self.error_test(x_train, x_valid)

        for i in range(iterations):
            self.params_update(params_dict, best_params)

            _, x_train, x_valid, _ = self.split_results(signal)
            error = self.error_test(x_train, x_valid)

            error_diff = best_error - error
            acceptable = np.exp(error_diff/(initial_temp-i))

            if error_diff < 0 or random() < acceptable:
                best_error = error
                best_params = [param[i] for param in params_dict]

        # return Params and Error
        return best_error

    @staticmethod
    def params_update(params_dict, best_params):
        for param in params_dict:
            param[0] = best_params[0] + randrange(param[1, 2]) * param[3]

    def error_test(self, train, compare) -> float:
        w_out = self.model.RidgeRegression(train, self.y_train)
        pred = compare @ w_out
        return torch.sum((pred - self.y_valid) ** 2) / len(self.y_valid)

    def split_results(self, signal, splits=None):
        if splits is None:
            splits = [250, 3750, 500, 500]
        if sum(splits) == len(signal):
            state = self.model.run(signal)
            return torch.split(state, splits)
        else:
            raise Exception("Splits do not add to same length as signal")
