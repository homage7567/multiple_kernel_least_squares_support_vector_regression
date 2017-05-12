import lssvm
import numpy as np
import pandas as pd
from cross_validation import CV
from matplotlib import pyplot as plt


def f(x):
    quad = x**2
    result = x + 4*np.exp(-2 * quad) / np.sqrt(2 * np.pi)
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
    Y = data["y"]
    X = data.drop("y", axis=1)

    kernel = lssvm.Kernel("gauss", [1.0])
    classifier = lssvm.LSSVMRegression(kernel, max_iter=10, c=50.0)

    x, y = CV.cross_val_score(data.drop("y", axis=1), data.drop("x", axis=1), classifier, segment_cnt=10)
    print("MSE = " + str(classifier.calculate_mse(x, y, lambda arg: f(arg))))
    plot_f(x, y)

if __name__ == '__main__':
    main()
