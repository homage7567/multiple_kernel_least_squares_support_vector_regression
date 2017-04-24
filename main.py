import lssvm
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt


def f(x):
    quad = x**2
    result = x + 4*np.exp(-2 * quad) / np.sqrt(2 * np.pi)
    return result


def plot_f(X, Y):
    y = []
    x = np.arange(-4.0, 4.1, 0.1)
    for xx in x:
        y.append(f(xx))

    plt.plot(x, y, label="y")
    # for i in range(X.size):
    #     elem = (X.iloc[i])["x"]
    #     y.append(f(elem))

    # plt.plot(X, y, label="y")
    plt.plot(X, Y, "g--", label="m(x)")
    # plt.legend(loc='upper center')

    plt.show()


def main():
    data = pd.read_excel('test_data.xlsx', header=0)
    Y = data["y"]
    X = data.drop("y", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=0)

    kernel = lssvm.Kernel("gauss", [1.0])
    classifier = lssvm.LSSVMRegression(kernel, c=50.0)
    classifier.fit(X_train, y_train, X_train.size)
    result = classifier.predict(X_test, X_test.size)
    plot_f(X_test, result)

if __name__ == '__main__':
    main()
