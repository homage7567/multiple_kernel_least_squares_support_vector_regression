import numpy as np
import numpy.linalg as lalg
from timer import Timer
from scipy.optimize import minimize

class LSSVMRegression(object):
    __X_train = 0
    __kernels = 0
    __e = 0
    __max_iter = 0
    __error_param = 0
    __alpha = 0
    __b = 0
    __c = 0
    __betas = 0
    __kernel_cnt = 0

    def __init__(self, kernels, error_param=0.00001, max_iter=10, c=1.0):
        self.__kernels = kernels
        self.__error_param = error_param
        self.__max_iter = max_iter
        self.__c = c
        self.__kernel_cnt = len(kernels)
        self.__betas = np.zeros(self.__kernel_cnt, dtype=float)
        for i in range(self.__kernel_cnt):
            self.__betas[i] = 1.0 / self.__kernel_cnt

    def fit(self, X_train, Y_train):
        def __calculate_alpha_b():
            I = np.ones(n, dtype=float)
            H = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1):
                    k = 0.0
                    for d in range(self.__kernel_cnt):
                        k += self.__betas[d] * self.__kernels[d].K(X_train.iloc[i].T, X_train.iloc[j])
                    H[i, j], H[j, i] = k, k
                H[i, i] += 1.0 / self.__c

            Hinv = lalg.inv(H)
            y = np.array(Y_train)
            b = (I.T @ Hinv @ y) / (I.T @ Hinv @ I)
            alpha = Hinv @ (y - I * b)

            return alpha, b

        def __calculate_beta(betas):
            sum = 0.0
            for i in range(n):
                sum_d = 0.0
                for d in range(self.__kernel_cnt):
                    K = [self.__kernels[d].K(X_train.iloc[i], X_train.iloc[k]) for k in range(n)]
                    beta_K_alpha = betas[d] * np.dot(K, self.__alpha)
                    sum_d += beta_K_alpha
                sum += (Y_train.iloc[i] - sum_d - self.__b)**2
            sum += 1
            return sum

        def __minimize_beta():
            cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1.0})
            bnds = [(0.0, 1.0) for i in self.__betas]
            betas = minimize(__calculate_beta, self.__betas, method='SLSQP', bounds=bnds, constraints=cons)
            return betas.x, betas.fun

        n = len(X_train)
        self.__X_train = X_train
        cur_iter = 0

        prev_beta_norm = np.linalg.norm(self.__betas)
        prev_func_value_norm = 0

        while cur_iter < self.__max_iter:
            print("Iteration " + str(cur_iter + 1))

            with Timer("Alphas estimation"):
                self.__alpha, self.__b = __calculate_alpha_b()

            with Timer("Betas estimation"):
                self.__betas, func_value = __minimize_beta()
            print("Betas: " + str(self.__betas))

            beta_norm = np.linalg.norm(self.__betas)
            func_value_norm = np.linalg.norm(func_value)
            beta_delta = abs(prev_beta_norm - beta_norm) < self.__error_param
            func_value_delta = abs(prev_func_value_norm - func_value_norm) < self.__error_param
            if beta_delta and func_value_delta: break

            prev_beta_norm = beta_norm
            prev_func_value_norm = func_value
            cur_iter += 1
            print("\n")
        return self.__alpha, self.__b

    def predict(self, X_test):
        def calculate_y(x):
            sum_j = 0.0
            for j in range(len(self.__X_train)):
                sum_d = 0.0
                xj = self.__X_train.iloc[j]
                for d in range(self.__kernel_cnt):
                    sum_d += self.__betas[d] * self.__kernels[d].K(x, xj)
                sum_j += self.__alpha[j] * sum_d
            return sum_j + self.__b

        y = [calculate_y(X_test.iloc[i]) for i in range(len(X_test))]
        return y


class Kernel(object):
    __kernel = 0
    __params = []
    __kernel_list = {
        'rbf': lambda sigma, x, xi, : np.e**(-lalg.norm(x - xi)**2 / (2 * sigma**2)),
        'linear': lambda x, xi: xi.T @ x,
        'poly': lambda c, d, x, xi: (1 + xi.T @ x / c)**d
    }

    def __init__(self, kernel, params):
        self.__kernel = self.__select_kernel(kernel)
        self.__params = params

    def __select_kernel(self, kenrel):
        """
        Выбор ядра из словаря
        :param kenrel: Название выбираемого ядра
        :return: Выбранное ядро
        """
        try:
            return self.__kernel_list[kenrel]
        except KeyError:
            raise ValueError("Select correct kernel! (example: 'gauss')")

    def K(self, *args):
        return self.__kernel(self.__params[0], *args)
