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
        _, self.y_train, self.y_valid, self.y_test = np.split(self._narma, [500, 4250, 4750])

    def grid_search(self):
        """Simple grid search implementation.
        """
        best_alpha = 0
        best_eta = 0
        best_err = 1
        best_gamma_all = 0
        for alpha in np.arange(0.05, 0.3, 0.01):
            for eta in np.arange(0.5, 1, 0.05):
                self.model.last_node.alpha = alpha
                self.model.last_node.eta = eta

                states = self.model.run(self.signal)  # need to pass alpha and eta to reservoir
                _, x_train, x_valid, x_test = np.split(states, [500, 3750, 500, 500])

                gammas = np.logspace(-9, -3, 7)
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
        """Class for parameter optimization options.

        Attributes
        ----------
        instance: object
            Instance of the object that contains the parameters to be optimised.
        name: str
            Parameters name.
        min: float = 0.1
            Bounding starting minimum value.
        max: float = 0.9
            Bounding starting maximum value.
        step: float = 0.05
            Value to be stepped per iteration.
        cur_val: float = uniform(min, max)
            Current values to be tested.
        best_val: float = cur_val
            Current best found values to minimize error.
        """
        instance: object
        name: str
        min: float = 0.1
        max: float = 0.9
        step: float = 0.05
        cur_val: float = uniform(min, max)
        best_val: float = cur_val

    def anneal(self, params_list: list[Param], iterations: int = 10, initial_temp=250):
        """Simulated Annealing for model hyperparameter optimisation.

        Tests multiple parameter options based on their min, max, and step values.
        Sets parameters to the best found during the algorithm.

        Parameters
        ----------
        params_list : list[Param]
            List of parameters to be optimised.
        iterations : int
            Number of iterations for annealing to perform.
        initial_temp : int

        Returns
        -------
        float
            Returns best error found
        """
        # Make first guess
        self.params_step(params_list)
        _, x_train, x_valid, x_test = self.split_results(self.signal)
        best_error = self.model.error_test(x_train, self.y_train, x_valid, self.y_valid)
        # For number of iterations make a guess
        for i in range(iterations):
            self.params_step(params_list)
            _, x_train, x_valid, x_test = self.split_results(self.signal)
            # Test the guess
            error = self.model.error_test(x_train, self.y_train, x_valid, self.y_valid)
            error_diff = best_error - error
            acceptable = np.exp(error_diff / (initial_temp - i))
            # If error is better or within acceptable range then accept as new best params
            if error_diff > 0 or random() > acceptable:
                best_error = error
                for param in params_list:
                    param.best_val = param.cur_val
        # Final update to ensure best found parameters
        for param in params_list:
            setattr(param.instance, param.name, param.best_val)
        # Return best error
        return best_error

    @staticmethod
    def params_step(params_list: list[Param]):
        """Steps the parameters for Simulated Annealing.

        Each step take the current best value and adds the product of a value inside the bounds
        and the step amount.

        Parameters
        ----------
        params_list : list[Param]
            List of parameters to be stepped.
        """
        for param in params_list:
            param.cur_val = param.best_val + uniform(param.min, param.max) * param.step
            setattr(param.instance, param.name, param.cur_val)

    def split_results(self, signal, splits=None):
        """Utility to split the output states

        Parameters
        ----------
        signal
            input signal for the model to run on
        splits
            size of state to split, default = [500, 4250, 4750]

        Returns
        -------
        list[ndarray[Any, dtype[_SCT]]]
            split states
        """
        if splits is None:
            splits = [500, 4250, 4750]
        state = self.model.run(signal)
        return np.split(state, splits)
