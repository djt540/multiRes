import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import DelayLine
import Reservoir


class Node(ABC):
    """Abstract class for nodes to inherit from.

    The node class template which enforces the use of a forward function
    to allow chaining of future nodes in the line and for use of the
    model class. Most classes here will require a wrapped parameter, however
    it is not required.

    Methods
    -------
    forward(self, signal: np.ndarray) -> np.ndarray


    Notes
    -----
    This allows chaining of nodes, with current implementations
    only allowing simplistic chaining (one directional line of nodes, other
    shapes have not been tested or properly implemented.)

    See Also
    --------
    Model : Model class - constructs and runs a model using Nodes
    DelayLine : A node subclass with wrapping functionality
    """
    @abstractmethod
    def forward(self, signal: np.ndarray) -> np.ndarray:
        """Abstract method for the Node classes forward function.
        Parameters

        Where modification of the incoming signal will take place
        before the signal is passed onto the next node (if the current
        node is wrapping another node).
        ----------
        signal : np.ndarray
            Input signal for the node.
        Returns
        -------
        np.ndarray
            Returns the output states
        """
        pass


class Model:
    def __init__(self, node_list: tuple):
        self.weights = None
        self.gamma = 1e-6
        self.node_list = node_list
        self.model_len = len(self.node_list)

        self.first_node = self.node_list[0]
        self.last_node = self.node_list[self.model_len - 1]
        self.num_nodes = self.last_node.num_nodes
        self.node_names = []

        for n in range(len(node_list) - 1):
            if hasattr(node_list[n], 'wrapped'):
                node_list[n].wrapped = node_list[n + 1]
            self.node_names.append(node_list[n].name)
        self.node_names.append(self.last_node.name)

    def run(self, signal: np.ndarray) -> np.ndarray:
        output = np.zeros((len(signal), self.num_nodes))
        for ts in range(len(signal) - 1):
            output[ts, :] = self.first_node.forward(signal[ts + 1])
        return output
    
    def simple_plot(self, prediction, target):
        plt.figure(figsize=(10, 5))
        plt.plot(prediction, label='Prediction')
        plt.plot(target, label='Target')
        plt.legend()
        plt.title(f'{self.__str__()}')
        plt.show()

    def __str__(self):
        return '->'.join(self.node_names)

    def ridge_regression(self, states, target):
        # Setup matrices from inputs
        mat_target = states.T @ target
        mat_states = states.T @ states
        # Perform ridge regression
        self.weights = np.linalg.pinv(mat_states + self.gamma * np.eye(len(mat_states))) @ mat_target
        return self.weights

    @staticmethod
    def NARMAGen(signal):
        ns = np.zeros((len(signal), 1))
        ns[0:10, 0] = signal[0:10]
        for t in range(len(ns) - 10):
            t += 10 - 1
            ns[t + 1, 0] = 0.3 * ns[t, 0] + 0.05 * ns[t, 0] * sum(ns[(t - (10 - 1)):t, 0]) + 1.5 * signal[t] * \
                signal[t - (10 - 1)] + 0.1
        return ns

    @staticmethod
    def NRMSE(pred, target):
        square_err = np.sum((pred - target) ** 2)
        var = np.var(target)
        return np.sqrt((square_err / var) * (1 / len(target)))

    def error_test(self, train, train_target, compare, compare_target):
        w_out = self.ridge_regression(train, train_target)
        pred = compare @ w_out
        return self.NRMSE(pred, compare_target)
