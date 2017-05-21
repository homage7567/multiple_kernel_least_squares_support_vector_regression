import lssvm
import numpy as np
import pandas as pd
from timer import Timer
from cross_validation import CV
from matplotlib import pyplot as plt

# x + 4*np.e**(-2 * quad) / np.sqrt(2 * np.pi)


def main():
    A = 2
    sigma = 0.1
    omega = 0.1 * 2 * np.pi
    noise = 0.1
    x = np.arange(0.0, 30.0, 0.05)

    def f(x):
        return A * np.e ** (-sigma * x) * np.sin(omega * x)

    def create_data(filename, draw=False):
        def generate_data(n_outliers=0):
            y = f(x)
            # rnd = np.random.RandomState(random_state)
            # error = noise * rnd.randn(len(x))
            # outliers = rnd.randint(0, len(x), n_outliers)
            # error[outliers] *= 35
            return y  # + error

        def plot_data():
            plt.plot(x, y, label='data')
            plt.legend()
            plt.show()

        y = generate_data(n_outliers=4)
        if draw: plot_data()
        df = pd.DataFrame({
            'x': x,
            "y": y})
        df.to_excel(filename)

    def classification():
        data = pd.read_excel('test1.xlsx', header=0)
        kernel_list = [lssvm.Kernel("gauss", [i]) for i in [0.1, 0.2, 0.3, 0.4]]
        classifier = lssvm.LSSVMRegression(kernel_list, c=5.0)

        with Timer("Cross Validaton"):
            x_avg, y_avg = CV.cross_val_score(data.drop("y", axis=1), data.drop("x", axis=1), classifier, segment_cnt=5)

        mse = classifier.calculate_mse(x_avg, y_avg, lambda arg: f(arg))
        print("MSE = " + str(mse))
        plot_f(x_avg, y_avg)

    def plot_f(X, Y):
        y = []
        for xx in x:
            y.append(f(xx))
        plt.plot(x, y, label="y")
        for a, b in zip(X, Y):
            plt.scatter(a, b)
        plt.show()

    # create_data("test1.xlsx", draw=True)
    classification()

if __name__ == '__main__':
    main()
