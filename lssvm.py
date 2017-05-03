import numpy as np
import numpy.linalg as lalg
import pandas as pd
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

    def __init__(self, kernels, error_param=0.0001, max_iter=15, c=1.0):
        self.__kernels = kernels
        self.__error_param = error_param
        self.__max_iter = max_iter
        self.__c = c
        self.__kernel_cnt = len(kernels)
        self.__betas = np.zeros(self.__kernel_cnt, dtype=float)
        for i in range(self.__kernel_cnt):
            self.__betas[i] = 1.0 / self.__kernel_cnt

    def cross_validation(self, X, Y, segment_cnt=2):
        result = []
        size = len(X) // segment_cnt
        # Первый блок
        x_test = X["x"][0:size]
        x_train = X["x"][size:]
        y_train = Y["y"][size:]
        print("Starting block 1: train_len = " + str(len(x_train)) + "; test_len = " + str(len(x_test)))
        self.fit(x_train, y_train)
        pred = self.predict(x_test)
        for elem in pred:
            result.append(elem)
        print("Block 1 complete!\n")

        # Внутренние блоки
        for i in range(1, segment_cnt - 1):
            x_train = X["x"][:i*size]
            y_train = Y["y"][:i * size]
            x_test = X["x"][i * size:(i + 1)*size]
            x_train = pd.concat([x_train, X["x"][(i + 1)*size:]], axis=0, ignore_index=True)
            y_train = pd.concat([y_train, Y["y"][(i + 1)*size:]], axis=0, ignore_index=True)
            print("Starting block " + str(i + 1) + ": train_len = " + str(len(x_train)) +
                  "; test_len = " + str(len(x_test)))
            self.fit(x_train, y_train)
            pred = self.predict(x_test)
            for elem in pred:
                result.append(elem)
            print("Block " + str(i + 1) + " complete!\n")

        # Последний блок
        x_test = X["x"][(segment_cnt - 1)*size:]
        x_train = X["x"][:(segment_cnt - 1)*size]
        y_train = Y["y"][:(segment_cnt - 1)*size]
        print("Starting block " + str(size) + " : train_len = " + str(len(x_train)) +
              "; test_len = " + str(len(x_test)))
        self.fit(x_train, y_train)
        pred = self.predict(x_test)
        for elem in pred:
            result.append(elem)
        print("Block " + str(size) + " complete!\n")
        return result

    def fit(self, X_train, Y_train):
        n = len(X_train)
        self.__X_train = X_train
        I = np.ones(n, dtype=float)
        H = np.zeros((n, n), dtype=float)
        cur_iter = 0

        def __calculate_alpha_b():
            # TODO: Проблема, что делать с кучей ядер??
            for i in range(n):
                for j in range(i + 1):
                    k = 0.0
                    for d in range(self.__kernel_cnt):
                        k += self.__betas[d]*self.__kernels[d].K(X_train.iloc[i], X_train.iloc[j])
                    H[i, j], H[j, i] = k, k
                H[i, i] += 1.0 / self.__c

            Hinv = lalg.inv(H)
            I_Hinv = np.matmul(I, Hinv)
            I_Hinv_Y = np.matmul(I_Hinv, Y_train)
            I_Hinv_I = np.matmul(I_Hinv, I)
            b = I_Hinv_Y / I_Hinv_I
            alpha = np.matmul(Hinv, Y_train - I * b)
            return alpha, b

        def __calculate_beta(betas):
            sum = 0.0
            beta_k = np.zeros(n, dtype=float)

            for i in range(n):
                for d in range(self.__kernel_cnt):
                    Kid = [betas[d] * self.__kernels[d].K(X_train.iloc[i], X_train.iloc[k]) for k in range(n)]
                    beta_k += Kid
                betta_k_alpha = np.matmul(beta_k, self.__alpha)
                sum += (Y_train.iloc[i] - betta_k_alpha - self.__b)**2
            sum += 1
            return sum

        def __minimize_beta():
            cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1.0})
            bnds = [(0.0, 1.0) for i in self.__betas]
            betaopt = minimize(__calculate_beta, self.__betas, method='SLSQP', bounds=bnds, constraints=cons)
            return betaopt.x

        prev_beta_norm = np.linalg.norm(self.__betas)
        while cur_iter < self.__max_iter:
            self.__alpha, self.__b = __calculate_alpha_b()

            print(self.__betas)
            self.__betas = __minimize_beta()
            beta_norm = np.linalg.norm(self.__betas)
            cur_iter += 1
            if abs(prev_beta_norm - beta_norm) < self.__error_param:
                break
        return self.__alpha, self.__b

    def predict(self, X_test):
        def calculate_y(xi):
            sum_j = 0.0
            for j in range(len(self.__X_train)):
                sum_d = 0.0
                xj = self.__X_train.iloc[j]
                for d in range(self.__kernel_cnt):
                    sum_d += self.__betas[d] * self.__kernels[d].K(xi, xj)
                sum_j += self.__alpha[j] * sum_d
            return sum_j + self.__b

        y = [calculate_y(X_test.iloc[i]) for i in range(len(X_test))]
        return y


class Kernel(object):
    __kernel = 0
    __params = []
    __kernel_list = {
        'gauss': lambda sigma, xi, xj, : np.e**(-1.0 * lalg.norm(xi - xj)**2 / (2 * sigma**2))
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

    def K(self, xi, xj):
        return self.__kernel(self.__params[0], xi, xj)
