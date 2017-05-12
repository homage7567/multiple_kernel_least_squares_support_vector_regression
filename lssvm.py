import numpy as np
import numpy.linalg as lalg
import pandas as pd


class LSSVMRegression(object):
    __X_train = 0
    __kernel = 0
    __e = 0
    __max_iter = 0
    __error_param = 0
    __alpha = 0
    __b = 0
    __c = 0

    def __init__(self, kernel, error_param=0.01, max_iter=5, c=1.0):
        self.__kernel = kernel
        self.__error_param = error_param
        self.__max_iter = max_iter
        self.__c = c

    def fit(self, X_train, Y_train):
        n = len(X_train)
        self.__X_train = X_train
        I = np.ones(n, dtype=float)
        H = np.zeros((n, n), dtype=float)
        cur_iter = 0

        def __calculate_alpha_b():
            for i in range(n):
                for j in range(0, i + 1):
                    k = self.__kernel.K(X_train.iloc[i], X_train.iloc[j])
                    H[i, j], H[j, i] = k, k
                H[i, i] += 1.0 / self.__c

            Hinv = lalg.inv(H)
            I_Hinv = np.matmul(I, Hinv)
            I_Hinv_Y = np.matmul(I_Hinv, Y_train)
            I_Hinv_I = np.matmul(I_Hinv, I)
            b = I_Hinv_Y / I_Hinv_I
            alpha = np.matmul(Hinv, Y_train - I * b)
            return alpha, b

        while cur_iter < self.__max_iter:
            self.__alpha, self.__b = __calculate_alpha_b()
            cur_iter += 1

        return self.__alpha, self.__b

    def predict(self, X_test):
        def calculate_y(xi):
            sum = 0.0
            for j in range(len(self.__X_train)):
                xj = self.__X_train.iloc[j]
                sum += self.__alpha[j] * self.__kernel.K(xi, xj)
            return sum + self.__b

        y = [calculate_y(X_test.iloc[i]) for i in range(len(X_test))]
        return y

    def calculate_mse(self, X, Y, f):
        mse = 0.0
        n = len(X)
        for i in range(n): mse += (f(X[i]) - Y[i]) ** 2
        mse /= n
        return np.sqrt(mse)

class Kernel(object):
    __kernel = 0
    __params = []
    __kernel_list = {
        'gauss': lambda sigma, xi, xj, : np.exp(-1.0 * lalg.norm(xi - xj)**2 / (2 * sigma**2))
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
