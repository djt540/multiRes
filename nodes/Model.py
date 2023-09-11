import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import TypeVar, Type, Tuple


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
    r"""The Model class handles constructing and running models based on a list of nodes.

    The model class will run through the list of nodes and string their 'wrapped' parameter
    allowing their forward functions to chain together into a completed system.

    The run method will run the constructed model using a given signal in, storing and returning
    the output states (stored as states in the model object). These states are later used
    for ridge regression learning for the model.

    Notes
    _____

    NARMA is the test I was using for all the models I created - so there is a NARMA
    function inside the model class, however this should really be in its own set of
    utility files along with the error calculations and plotting functions.

    Error calculation here means normalised root-mean-square error, as seen in the following
    formula:
    .. math::
        \sqrt{\frac{1}{M}\frac{\sum_{k=1}^{M}(\hat{y_k}-y_k)^2}{\sigma ^2(y_k)}}

    found in the supplementary information from the single dynamical node paper [SDN]_

    The error calc function call the ridge regression function as well based on the
    states and target provided.

    Parameters
    ----------
    node_list : tuple['Node', ...]
        List of nodes to be made into a model. The first node in the tuple will be the first
        node to receive the signal, with the last one receiving the signal last.

    Attributes
    ----------
    states : np.ndarray
        Output of the model. Used for Ridge Regression.
    gamma : float
        Parameter used in ridge regression.
    node_list : tuple[Node]
        List of nodes in the model, in order.
    model_len : int
        Number of node objects in the model (however this currently doesnt add the number of nodes
        hidden inside a NodeArray).
    first_node : Node
        First node in the model.
    last_node : Node
        Last node in the model.
    num_nodes : int
        Number of nodes in the model (size of all spilt signals added together)
        e.g. a model with 3 ESN's each 100 nodes in size will have a num nodes of 300,
        a model with 1 ESN with 100 nodes will have a num_nodes of 100.
    node_names : list[str]
        List of the nodes names, these are all set in the node classes init and
        required to create a model.

    Raises
    ------
    IndexError
        If there are not enough nodes in the list to create a model (minimum 2 nodes).
    AttributeError
        If the last node doesnt have a num_nodes attribute, e.g. a delay line is the final node.
    AttributeError
        If not all the nodes in the list have a name.

    See Also
    --------
    Node : Node class required to form a model.
    """
    def __init__(self, node_list: tuple['Node', ...]):
        self.states = None
        self.gamma = 1e-6
        self.node_list = node_list
        self.model_len = len(self.node_list)

        try:
            self.first_node = self.node_list[0]
            self.last_node = self.node_list[self.model_len - 1]
        except IndexError:
            print("Not enough nodes in the list")
            raise
        try:
            self.num_nodes = self.last_node.num_nodes
        except AttributeError:
            print("The last node does not have any neurons/num_nodes empty")
            raise
        try:
            self.node_names = [node.name for node in node_list]
        except AttributeError:
            print("Not all nodes in the model have a name, check your subclass")
            raise

        for n in range(len(node_list) - 1):
            if hasattr(node_list[n], 'wrapped'):
                node_list[n].wrapped = node_list[n + 1]

    def run(self, signal: np.ndarray) -> np.ndarray:
        """Run function for the model.

        This handles the run loop for the model, which is the length of the signal provided.
        An output of the signals length and the depth of the number of total nodes[#num_nodes]_ in the
        model is created. The output is the state of the model for each timestep.

        Run simply passes the signal to the forward function of the first node in the
        node_list and would need to be modified if a more complex model class was created[#fut]_.

        .. rubric:: Footnotes

        .. [#num_nodes] I know this is slightly confusing naming but here it means num_nodes as explained
        in the model class documentation.

        .. [#fut] For example if a Model class allowed for multiple inputs or outputs, however
        this is beyond the scope of this project.

        Parameters
        ----------
        signal: np.ndarray
            Signal to run the model, for NARMA this will be an array of random noise.
        Returns
        -------
        np.ndarray
            Output states.
        """
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
        self.states = np.linalg.pinv(mat_states + self.gamma * np.eye(len(mat_states))) @ mat_target
        return self.states

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
