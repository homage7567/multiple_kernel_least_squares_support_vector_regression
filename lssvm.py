import numpy as np
import pandas as pd
import numpy.linalg as lalg
from numpy.random import normal


class LSSVMRegression(object):
    __kernel = 0
    __X = 0
    __Y = 0
    __n = 0
    __e = 0
    __error_param = 0

    __kernel_list = {
        'gauss': lambda x, z, sigma: np.exp(- lalg.norm(x - z) ** 2 / sigma)
    }

    def __init__(self, error_param=0.01, kernel_name='gauss'):
        """
        Конструктор
        :param error_param: Параметр независимой и одинаково распределённой ошибка
        :param kernel_name: Название ядра
        """
        self.__kernel = self.__select_kernel(kernel_name)
        self.__error_param = error_param

    def __select_kernel(self, kenrel_name):
        """
        Выбор ядра из словаря
        :param kenrel_name: Название выбираемого ядра
        :return: Выбранное ядро
        """
        try:
            return self.__kernel_list[kenrel_name]
        except KeyError:
            raise ValueError("Select correct kernel! (example: 'gauss')")

    def leave_one_out(self):


    def search_aproximating_func(self, X, Y):
        self.__e = normal(0, self.__error_param, self.__n)


class Model(object):
    __function = 0
    X = 0
    Y = 0
    data = 0

    def read_from_excel(self, excel_name):
        self.data = pd.read_excel(excel_name, header=0)
        self.Y = self.data["y"]
        self.X = self.data["x"]

    def create_data(self, X, function):
        self.__function = function
        self.X = X
        self.Y = np.zeros(len(X))
        for i in range(len(self.Y)):
            self.Y[i] = self.__function(X[i])
