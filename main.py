import lssvm
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
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
    Y = data["y"]
    X = data.drop("y", axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.4, random_state=0)

    kernel_list = [lssvm.Kernel("gauss", [i]) for i in [.5, 0.1, 1.0, 0.0]]
    classifier = lssvm.LSSVMRegression(kernel_list, c=50.0)

    result = classifier.cross_validation(data.drop("y", axis=1), data.drop("x", axis=1), segment_cnt=10)
    plot_f(data["x"], result)

if __name__ == '__main__':
    main()
