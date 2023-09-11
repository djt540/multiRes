from nodes.Rotor import Rotor
from nodes.DelayLine import DelayLine
from nodes.Reservoir import Reservoir, InputMask
from nodes.Model import *
from util.ParamOptim import ParamOpt
from nodes.NodeArray import NodeArray


def error_average(model_desc, num_tests):
    errors = []
    template_mod = Model(model_desc)

    for x in range(num_tests):
        sig = np.random.rand(5250) / 2
        mod = Model(model_desc)
        po = ParamOpt(mod, sig)
        errors.append(po.grid_search())

    # with open('test-results.txt', 'a') as f:
    #     for error in errors:
    #         f.write(f'{template_mod.node_list[1].fb_str}, {template_mod.node_list[1].tau}, {error}\n')

    avrg_error = sum(errors) / num_tests
    with open('util/test-results.csv', 'a') as f:
        f.write(f'{template_mod.node_list[1].fb_str}, {template_mod.node_list[1].tau}, {avrg_error}\n')


def fb_tau_tester(model_desc):
    sig = (np.random.rand(5250) / 2)

    mod = Model(model_desc)
    res = mod.last_node

    narma = mod.NARMAGen(sig)
    wash, y_train, y_valid, y_test = np.split(narma, [500, 4250, 4750])

    test_size = 20
    error_mat = np.ndarray((test_size, test_size))

    # sweep tau and fb
    for tm in range(test_size):
        # CHANGE THIS IF THE NODE CHANGES LOCATION IN THE MODEL
        mod.first_node.fb_str = 0.1 + (tm / test_size)
        for fb in range(test_size):
            res.reset_states()
            mod.first_node.eta = 0.1 + (fb / test_size)
            states = mod.run(sig)
            wash, x_train, x_valid, x_test = np.split(states, [500, 4250, 4750])

            w_out = mod.ridge_regression(x_train, y_train)
            pred = x_test @ w_out
            error_mat[tm][fb] = mod.NRMSE(pred, y_test)
    try:
        np.savetxt('util/test-results.csv', error_mat, delimiter=",", fmt='%f')
    except FileNotFoundError:
        print(error_mat)


def _tester(model_desc, multiRes=False):
    np.random.seed(seed=1)
    sig = (np.random.rand(5250) / 2)

    errors = []
    for i in range(1):
        mod = Model(model_desc)
        po = ParamOpt(mod, sig)

        narma = mod.NARMAGen(sig)
        wash, y_train, y_valid, y_test = np.split(narma, [500, 4250, 4750])

        if multiRes:
            # for multi ESN
            res = mod.last_node.nodes
            opt_params = [ParamOpt.Param(instance=res[0], name='_leak', min=0.6),
                          ParamOpt.Param(instance=res[0], name='_in_scale', min=0.05),
                          ParamOpt.Param(instance=res[1], name='_leak', min=0.6),
                          ParamOpt.Param(instance=res[1], name='_in_scale', min=0.05),
                          ParamOpt.Param(instance=res[2], name='_leak', min=0.6),
                          ParamOpt.Param(instance=res[2], name='_in_scale', min=0.05),
                          ParamOpt.Param(instance=res[3], name='_leak', min=0.6),
                          ParamOpt.Param(instance=res[3], name='_in_scale', min=0.05),
                          ParamOpt.Param(instance=res[4], name='_leak', min=0.6),
                          ParamOpt.Param(instance=res[4], name='_in_scale', min=0.05),
                          ]
            po.anneal(opt_params)

        # # for just one res
        # res = mod.last_node
        # del_line = mod.node_list[1]
        # opt_params = [ParamOpt.Param(instance=res, name='leak', min=0.6),
        #               ParamOpt.Param(instance=res, name='in_scale', min=0.05),
        #               ParamOpt.Param(instance=del_line, name='fb_str', min=0.5),
        #               ParamOpt.Param(instance=del_line, name='eta', min=0.1),
        #               ]
        #
        # po.anneal(opt_params)
        #
        # print(res)
        # res.reset_states()
        states = mod.run(sig)
        wash, x_train, x_valid, x_test = np.split(states, [500, 4250, 4750])
        w_out = mod.ridge_regression(x_train, y_train)
        pred = x_valid @ w_out
        print(mod.NRMSE(pred, y_valid))
        mod.simple_plot(pred, y_valid)


if __name__ == "__main__":
    nnodes = 400
    res_num = 10
    res = [Reservoir(nnodes) for i in range(res_num)]
    total_nodes = nnodes * res_num

    # # # Rotating Signal then Masking
    # _tester((InputMask(total_nodes), Rotor(res_num, total_nodes), NodeArray(res)))
    # for i in res:
    #     i.reset_states()

    # Rotor Wrapping Delay Line
    _tester((InputMask(total_nodes), Rotor(res_num, total_nodes), DelayLine(tau=18, fb_str=0.5), NodeArray(res)))

    # Delay Line wrapping Rotor
    # _tester((DelayLine(tau=20, fb_str=0.4), InputMask(total_nodes), Rotor(res_num, total_nodes), NodeArray(res)))
    # for i in res:
    #     i.reset_states()

    # This is delayline wrapping rotor
    # _tester((DelayLine(tau=80, fb_str=0.4, eta=0.2), Rotor(len(res), total_nodes), NodeArray(res)),
    #         multiRes=True)
    # for i in res:
    #     i.reset_states()

    # delay line
    # _tester((DelayLine(tau=84, fb_str=0.5), InputMask(nnodes), res[0]))
    # res[0].reset_states()
    #
    # # This is just single ESN - unfortunately using optimParams for the previous model
    # _tester((InputMask(nnodes), res[0]))
    #
    # fb_tau_tester((DelayLine(tau=20, fb_str=0.5), InputMask(nnodes), res[0]))
