import lssvm
import numpy as np
import pandas as pd
from cross_validation import CV
from matplotlib import pyplot as plt


def f(x):
    quad = x**2
    result = x + 4*np.e**(-2 * quad) / np.sqrt(2 * np.pi)
    return result


def plot_f(X, Y):
    y = []
    x = np.arange(-4.0, 4.0, 0.05)
    for xx in x:
        y.append(f(xx))
    plt.plot(x, y, label="y")
    for a, b in zip(X, Y):
        plt.scatter(a, b)
    plt.show()


def main():
    data = pd.read_excel('test_data.xlsx', header=0)
    kernel_list = [lssvm.Kernel("gauss", [i]) for i in [-0.5, -0.1, 0.1, 0.5]]
    classifier = lssvm.LSSVMRegression(kernel_list, c=20.0)
    x, y = CV.cross_val_score(data.drop("y", axis=1), data.drop("x", axis=1), classifier, segment_cnt=10)
    mse = classifier.calculate_mse(x, y, lambda arg: f(arg))
    print("MSE = " + str(mse))
    plot_f(x, y)

if __name__ == '__main__':
    main()
