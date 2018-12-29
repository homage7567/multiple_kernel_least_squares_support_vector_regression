import numpy as np
import numpy.linalg as lalg
from timer import Timer
from scipy.optimize import minimize
from statsmodels import robust


class MKLSSVR(object):
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
    __c_one = 0
    __c_two = 0

    def __init__(self, kernels, error_param=0.00001, max_iter=10, c=1.0, c_one=0.0, c_two=0.0):
        '''
        Инициализация класса для оценивания модели
        :param kernels: Набор ядер
        :param error_param: Точность вычислеинй
        :param max_iter: Максимальное количество итераций
        :param c: Параметр регуляризации
        '''
        self.__kernels = kernels
        self.__error_param = error_param
        self.__max_iter = max_iter
        self.__c = c
        self.__kernel_cnt = len(kernels)
        self.__c_one = c_one
        self.__c_two = c_two
        self.__betas = np.array(
            [1.0 / self.__kernel_cnt for _ in range(self.__kernel_cnt)])

    def fit(self, X_train, Y_train):
        '''
        Алгоритм обучения MK LS SVM по тренировочной выборке
        :param X_train: Набор аргументов из тренировочной выборки
        :param Y_train: Набор значений функции из тренировочной выборки
        :return: Параметры модели: альфа и значение смещения b
        '''
        def __calculate_alpha_b():
            '''
            Оценка параметров: альфа и значения смещения b
            :return: Оценённые параметры альфа и значения смещения b
            '''
            I = np.ones(n, dtype=float)
            H = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i, n):
                    k = 0.0
                    for d in range(self.__kernel_cnt):
                        k += self.__betas[d] * \
                            self.__kernels[d].K(X_train[i], X_train[j])
                    H[i, j], H[j, i] = k, k
                H[i, i] += 1.0 / self.__c

            Hinv = lalg.inv(H)
            y = np.array(Y_train)
            b = (I.T @ Hinv @ y) / (I.T @ Hinv @ I)
            alpha = Hinv @ (y - I * b)

            return alpha, b

        def __weight_calculate_alpha_b():
            V = np.zeros((n, n), dtype=float)
            I = np.ones(n, dtype=float)
            H = np.zeros((n, n), dtype=float)
            for i in range(n):
                ek = self.__alpha[i] / self.__c
                uk = __calculate_uk(ek, X_train[i])
                V[i][i] = 1 / (self.__c * uk)
                for j in range(i, n):
                    k = 0.0
                    for d in range(self.__kernel_cnt):
                        k += self.__betas[d] * \
                            self.__kernels[d].K(X_train[i], X_train[j])
                    H[i, j], H[j, i] = k, k

            Hinv = lalg.inv(H + V)
            y = np.array(Y_train)
            b_star = (I.T @ Hinv @ y) / (I.T @ Hinv @ I)
            alpha_star = Hinv @ (y - I * b_star)

            return alpha_star, b_star

        def mad(data, axis=None):
            return np.mean(np.abs(data - np.mean(data, axis)), axis)

        def __calculate_uk(ek, xi):
            uk = 0.0
            s = 1.483*robust.mad(xi)

            if abs(ek/s) <= self.__c_one:
                uk = 1.0
            elif self.__c_one <= abs(ek/s) <= self.__c_two:
                uk = (self.__c_two - abs(ek/s))/(self.__c_two - self.__c_one)
            else:
                 uk = 1e-4
            return uk

        def __minimize_beta():
            '''
            Решение задачи квадратичного программирования для нахождения оптимальные весовых коэффициентах при ядрах
            методов последовательного квадратичного программирования (SLSQP)
            :return:
            '''

            def __calculate_beta(betas):
                '''
                Функционал, который необходимо
                :param betas:
                :return:
                '''
                sum = 0.0
                for i in range(n):
                    sum_k = np.zeros(n, dtype=float)
                    K = np.zeros(n, dtype=float)
                    for d in range(self.__kernel_cnt):
                        for j in range(n):
                            tmp1 = X_train[i]
                            tmp2 = X_train[j]
                            tmp = self.__kernels[d].K(X_train[i], X_train[j])
                            K[j] = tmp
                        sum_k += betas[d] * K
                    sum += (Y_train[i] - np.dot(sum_k,
                                                self.__alpha) - self.__b)**2
                return sum

            prev_fun = __calculate_beta(self.__betas)

            cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1.0})
            bnds = [(0.0, 1.0) for _ in self.__betas]
            betas = minimize(__calculate_beta, self.__betas, method='SLSQP', bounds=bnds, constraints=cons,
                             options={'maxiter': 5000, 'disp': True})
            return betas.x, betas.fun, prev_fun

        n = len(X_train)
        self.__X_train = X_train
        cur_iter = 0

        prev_beta_norm = np.linalg.norm(self.__betas)
        prev_func_value_norm = 0

        while cur_iter < self.__max_iter:
            print("Iteration " + str(cur_iter + 1))

            with Timer("Alphas estimation"):
                self.__alpha, self.__b = __calculate_alpha_b()

            with Timer("Robust estimation"):
                self.__alpha, self.__b = __weight_calculate_alpha_b()

            if self.__kernel_cnt > 1:
                with Timer("Betas estimation"):
                    self.__betas, func_value, prev_fun = __minimize_beta()

                    str_beta = ""
                    for beta in self.__betas:
                        str_beta += '{:.3f} '.format(beta)
                    print("Betas: " + str_beta)
                    print("Func Value before optimization: " +
                          '{:.3f}'.format(prev_fun))
                    print("Func Value after optimization: " +
                          '{:.3f}'.format(func_value))

                beta_norm = np.linalg.norm(self.__betas)
                func_value_norm = np.linalg.norm(func_value)
                beta_delta = abs(prev_beta_norm -
                                 beta_norm) < self.__error_param
                func_value_delta = abs(
                    prev_func_value_norm - func_value_norm) < self.__error_param
                if beta_delta or func_value_delta:
                    break
                prev_beta_norm = beta_norm
                prev_func_value_norm = func_value

            cur_iter += 1
        return self.__alpha, self.__b, self.__betas

    def predict(self, X_test):
        '''
        Оценка значений по аргументам из обучающей выборки
        :param X_test:
        :return:
        '''
        def calculate_y(x):
            sum_j = 0.0
            for j in range(len(self.__X_train)):
                sum_d = 0.0
                xj = self.__X_train[j]
                for d in range(self.__kernel_cnt):
                    sum_d += self.__betas[d] * self.__kernels[d].K(x, xj)
                sum_j += self.__alpha[j] * sum_d
            return sum_j + self.__b

        y = [calculate_y(X_test[i]) for i in range(len(X_test))]
        return y

    def get_betas(self):
        return self.__betas

    def reset_alpha_beta_b(self):
        self.__betas = np.array(
            [1.0 / self.__kernel_cnt for _ in range(self.__kernel_cnt)])
        self.__alpha = 0
        self.__b = 0


class Kernel(object):
    '''
    Класс, определяющий работу с ядром
    '''
    __kernel = 0
    __kernel_name = ''
    __params = []
    __kernel_list = {
        'rbf': lambda sigma, x, xi, : np.e**(-lalg.norm(x - xi)**2 / (2 * sigma**2)),
        'linear': lambda x, xi: xi.T @ x,
        'poly': lambda c, d, x, xi: (1 + xi.T @ x / c)**d
    }

    def __init__(self, kernel, params):
        '''
        Инициализация класса для работы с ядром
        :param kernel: Строковый параметр, принимающий название ядра
        :param params: Параметры ядерной функции
        '''
        self.__kernel = self.__select_kernel(kernel)
        self.__params = params

    def __select_kernel(self, kernel):
        """
        Выбор ядра из списки ядер
        :param kernel: Название выбираемого ядра
        :return: Выбранное ядро
        """
        try:
            self.__kernel_name = kernel
            return self.__kernel_list[kernel]
        except KeyError:
            raise ValueError("Select correct kernel! (example: 'gauss')")

    def K(self, *args):
        '''
        Выполнения ядерного преборазования
        :param args: Аргументы, передаваемые в ядерную функцию
        :return: Результат ядерного преобразования
        '''
        if self.__kernel_name == "rbf":
            return self.__kernel(self.__params[0], *args)
        if self.__kernel_name == "linear":
            return self.__kernel(*args)
        if self.__kernel_name == "poly":
            return self.__kernel(self.__params[0], self.__params[1], *args)
