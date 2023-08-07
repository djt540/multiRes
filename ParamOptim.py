from Model import *


class ParamOpt:
    def __init__(self, model: Model):
        self.model = model
        self.num_nodes = self.model.last_node.num_nodes

    def run(self, signal: torch.Tensor):
        narma = self.model.NARMAGen(signal, 10)
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
                _, y_train, y_valid, y_test = torch.split(narma, [250, 3750, 500, 500])

                gammas = torch.logspace(-9, -3, 7)
                best_gamma = 0
                local_best_error = 1
                for gamma in gammas:
                    w_out = self.model.RidgeRegression(x_train, y_train, gamma)
                    y_pred = x_valid @ w_out
                    error = torch.sum((y_pred - y_valid) ** 2) / len(y_valid)

                    if error < local_best_error:
                        best_gamma = gamma
                        local_best_error = error

                w_out = self.model.RidgeRegression(x_train, y_train, best_gamma)
                y_pred = x_test @ w_out

                error = torch.sum((y_pred - y_valid) ** 2) / len(y_valid)

                if error < best_err:
                    best_alpha = alpha
                    best_eta = eta
                    best_err = error
                    best_gamma_all = best_gamma

        # print(f"Alpha:{best_alpha} Eta:{best_eta} Gamma:{best_gamma_all}")
        # print(f"best error:{best_err}")

        self.model.last_node.alpha = best_alpha
        self.model.last_node.eta = best_eta
        states = self.model.run(signal)

        _, x_train, x_valid, x_test = torch.split(states, [250, 3750, 500, 500])
        _, y_train, y_valid, y_test = torch.split(narma, [250, 3750, 500, 500])
        w_out = self.model.RidgeRegression(x_train, y_train, best_gamma_all)

        # prediction = x_valid @ w_out

        # self.model.simple_plot(x_valid, y_valid)
        return best_err
