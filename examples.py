import numpy as np

from nodes.Rotor import Rotor
from nodes.DelayLine import DelayLine
from nodes.Reservoir import Reservoir, InputMask
from nodes.NodeArray import NodeArray
from nodes.Model import Model
from util.ParamOptim import ParamOpt


def fb_tau_tester(model: Model):
    """
    Creates a matrix of errors based on varying both feedback and input scaling
    in a delayline based model.

    For simplicity this example only works when the delayline is the first node
    in the model.

    Results will be output in a file in utils called test-results.csv and can
    be turned into a heatmap in DataAnalysis.py
    """
    # -------------------------------Run Variables------------------------------- #
    sig = (np.random.rand(5250) / 2)
    res = model.last_node
    narma = model.NARMAGen(sig)
    wash, y_train, y_valid, y_test = np.split(narma, [500, 4250, 4750])
    # -------------------------------Test Variables------------------------------ #
    test_size = 20
    error_mat = np.ndarray((test_size, test_size))
    # --------------------------------------------------------------------------- #
    # Create the file
    f = open("util/test-results.csv", "a")
    f.close()
    # Run the sweeps for the test
    for fb in range(test_size):
        # Location of the delay node should be first in the list for this test
        # Set new feedback strength
        model.first_node.fb_str = 0.1 + (fb / test_size)
        for eta in range(test_size):
            # Set new feedback input scaling
            model.first_node.eta = 0.1 + (eta / test_size)
            # Reset the models current state
            res.reset_states()
            # Run the model
            states = model.run(sig)
            wash, x_train, x_valid, x_test = np.split(states, [500, 4250, 4750])
            # Train model and calculate error
            w_out = model.ridge_regression(x_train, y_train)
            pred = x_test @ w_out
            # Add error to error matrix
            error_mat[fb][eta] = model.NRMSE(pred, y_test)
    # Try to save the error matrix to a file, if anything goes wrong print the error matrix
    try:
        np.savetxt('util/test-results.csv', error_mat, delimiter=",", fmt='%f')
    except Exception as e:
        print(error_mat)
        print(e)


def res_optim(model: Model, res_list):
    """Demonstrates optimisation of models that rely on multiple reservoirs

    Optimises the leak rate and input scaling for a model containing multiple
    reservoirs, it uses the paramOptim simulated annealing method with verbose
    set to true to allow you to see error decrease (hopefully).

    There are some internesting parameters to mess around with here so looking
    at and changing the code for the simulated annealing can provide very
    results. For example changing the type of distributions can produce
    varying results.
    Parameters
    ----------
    model:
        The model that is being optimised
    res_list
        The list of reservoir instances to be optimised.

    """
    # Create input signal
    sig = (np.random.rand(5250) / 2)
    # Create list of parameters based on the reservoir list
    opt_params = [ParamOpt.Param(instance=r, name='_leak', min=0.6, step=0.01)
                  for r in res_list] + \
                 [ParamOpt.Param(instance=r, name='_in_scale', min=0.1, step=0.01)
                  for r in res_list]
    # Create optimiser instance
    po = ParamOpt(model, sig)
    # Run optimiser
    po.anneal(opt_params, verbose=True)


def test_model(model: Model):
    """A simple function to demonstrate testing of unoptimised models

    Demonstration of a model, with its NRMSE NARMA score as well as plotting
    the output from the model.
    """
    # -------------------------------Run Variables------------------------------- #
    sig = (np.random.rand(5250) / 2)
    narma = model.NARMAGen(sig)
    wash, y_train, y_valid, y_test = np.split(narma, [500, 4250, 4750])
    # -------------------------------Test Variables------------------------------ #
    states = model.run(sig)
    wash, x_train, x_valid, x_test = np.split(states, [500, 4250, 4750])
    w_out = model.ridge_regression(x_train, y_train)
    pred = x_valid @ w_out
    print(model.NRMSE(pred, y_valid))
    model.simple_plot(pred, y_valid)


if __name__ == '__main__':
    """Setup of and running the examples
    """
    # number of nodes in each reservoir
    nnodes = 100
    # number of reservoirs in the res list
    res_num = 1
    # creation of reservoirs
    res = [Reservoir(nnodes) for i in range(res_num)]
    # calculating the total number of nodes in the model
    total_nodes = nnodes * res_num

    # setup of ESN
    ESN = Model((InputMask(nnodes),
                 res[0]))

    # setup of Rotor
    rot = Model((InputMask(total_nodes),
                 Rotor(res_num, total_nodes),
                 NodeArray(res)))

    # setup of delay line
    delay_line = Model((DelayLine(tau=84, fb_str=0.5),
                        InputMask(nnodes),
                        res[0]))

    # setup of rotor wrapping a delay line
    rot_del = Model((InputMask(total_nodes),
                     Rotor(res_num, total_nodes),
                     DelayLine(tau=18, fb_str=0.5),
                     NodeArray(res)))

    # setup of delay line wrapping a rotor
    del_rot = Model((DelayLine(tau=20, fb_str=0.4),
                     InputMask(total_nodes),
                     Rotor(res_num, total_nodes),
                     NodeArray(res)))

    # # Function examples, uncomment to use # #
    # fb_tau_tester(delay_line)
    # res_optim(ESN, res)
    # test_model(rot)
