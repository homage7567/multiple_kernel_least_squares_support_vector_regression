import sklearn.svm


class LSSVMRegression(object):
    __kernel = 0

    def __init__(self, kernel="gauss"):
        self.__kernel = kernel