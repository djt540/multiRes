import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Node(ABC):
    @abstractmethod
    def forward(self, signal: torch.Tensor, fb_str: float = 0) -> torch.Tensor:
        pass


class Model:
    def __init__(self, node_list: tuple):
        self.weights = None
        self.gamma = 0.3
        self.node_list = node_list
        self.model_len = len(self.node_list)

        self.first_node = self.node_list[0]
        self.last_node = self.node_list[self.model_len - 1]
        self.node_names = []

        for n in range(len(node_list) - 1):
            node_list[n].wrapped = node_list[n + 1]
            self.node_names.append(node_list[n].name)
        self.node_names.append('Res')

    def run(self, signal: torch.Tensor) -> torch.Tensor:
        output = torch.zeros((len(signal), self.last_node.num_nodes))
        for ts in range(len(signal)):
            output[ts, :] = self.first_node.forward(signal[ts])
        return output
    
    def simple_plot(self, testing, target):
        prediction = testing @ self.weights
        plt.figure(figsize=(10, 5))
        plt.plot(prediction, label='Prediction')
        plt.plot(target, label='Target')
        plt.legend()
        plt.title = f'{self.__str__()}'
        plt.show()

    def __str__(self):
        return '->'.join(self.node_names)

    def ridge_regression(self, states, target):
        # Setup matrices from inputs
        mat_target = states.T @ target
        mat_states = states.T @ states
        # Perform ridge regression
        self.weights = torch.linalg.pinv(mat_states + self.gamma * torch.eye(len(mat_states))) @ mat_target
        return self.weights

    @staticmethod
    def NARMAGen(signal, N):
        ns = torch.zeros((len(signal), 1))
        ns[0:N, 0] = signal[0:N]
        for t in range(len(ns) - N):
            t += N - 1
            ns[t + 1, 0] = 0.3 * ns[t, 0] + 0.05 * ns[t, 0] * sum(ns[(t - (N - 1)):t, 0]) + 1.5 * signal[t] * \
                signal[t - (N - 1)] + 0.1
        return ns

    @staticmethod
    def NRMSE(pred, target):
        square_err = torch.sum((pred - target) ** 2) / len(target)
        var = torch.var(target)
        return torch.sqrt(square_err / var)

    def error_test(self, train, train_target, compare, compare_target):
        w_out = self.ridge_regression(train, train_target)
        pred = compare @ w_out
        return self.NRMSE(pred, compare_target)
