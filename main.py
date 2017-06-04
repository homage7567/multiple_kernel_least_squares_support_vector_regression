import sys
import lssvm
import numpy as np
import pandas as pd
from timer import Timer
import multiprocessing as mp
from cross_validation import CV
from matplotlib import pyplot as plt


def f(x):
    # return 2 * np.e**(-0.1 * x) * np.sin(0.1 * 2 * np.pi * x)
    return np.sinc(0.3*x)


def plot_f(x, y, x_est, y_est, file_out, dpi=100, figsize=(32, 16)):
    data = sorted(zip(x_est, y_est))
    x_ = [x[0] for x in data]
    y_ = [y[1] for y in data]
    y_cur = f(np.asarray(x_))

    plt.figure(dpi=dpi, figsize=figsize)
    plt.plot(x, y, 'bo', label="Data", alpha=0.5)
    plt.plot(x, y_cur, 'k', label="Real function")
    plt.plot(x_, y_, "g-", label="Estimated function")
    plt.legend()
    plt.savefig(file_out + ".jpeg")


def one_research(data, kernel_params, reg_param):
    file_out = "Research\\" + str(kernel_params) + ", " + str(reg_param)
    sys.stdout = open(file_out + ".txt", "w+")
    kernel_list = [lssvm.Kernel("rbf", [kernel_param]) for kernel_param in kernel_params]
    classifier = lssvm.LSSVMRegression(kernel_list, c=reg_param)

    with Timer("Cross Validation"):
        cv_score = CV.cross_val_score(data["x"], data["y"], classifier, f, segment_cnt=5)

    with Timer("Fit"):
        classifier.fit(data["x"], data["y"])
        y_est = classifier.predict(data["x"])

    mse = CV.calculate_mse(data["x"].tolist(), y_est, lambda arg: f(arg))

    print("CV score = " + str(cv_score))
    print("MSE = " + str(mse))
    plot_f(data["x"], data["y"], data["x"].tolist(), y_est, file_out)
    return mse, cv_score


def function_estimation(data, mutex, *kernels):
    kernel_params = [k for k in kernels]
    for r in range(1, 100, 10):
        r /= 1.0
        mse, cv = one_research(data, kernel_params, r)
        with mutex:
            report_file = open("Research\\report.txt", "a")
            report_file.write(str(mse) + "\t" + str(cv) + "\t" + str(r) + "\t" + str(kernel_params) + "\n")
            report_file.close()
    return


def main():
    def create_data(filename, draw=False):
        def generate_data(n_outliers=0):
            y = f(x)
            rnd = np.random.RandomState()
            error = noise * rnd.rand(len(x))
            sign = [-1, 1]
            for i in range(len(error)):
                error[i] *= sign[rnd.random_integers(0, 1)]
            outliers = rnd.randint(0, len(x), n_outliers)
            error[outliers] *= 1
            return y + error

        def plot_data():
            y_cur = [f(i) for i in x]
            plt.plot(x, y_cur, 'r--', label="Real Function")
            plt.plot(x, y, 'bo', label='Data', alpha=0.5)
            plt.legend()
            plt.show()

        y = generate_data(n_outliers=4)
        if draw: plot_data()
        df = pd.DataFrame({
            'x': x,
            "y": y})
        df.to_excel(filename)

    def researches(data):
        report_file = open("Research\\report.txt", "w+")
        report_file.write("MSE\tCV\tReg_param\tKernel_param\n")
        report_file.close()
        kp = [i / 100.0 for i in range(100, 600, 30)]
        i = 0
        thread_list = []
        mutex = mp.Lock()

        with Timer("Time with threads"):
            while i < len(kp) - 4:
                thread = mp.Process(target=function_estimation,
                                    args=(data, mutex, kp[i], kp[i + 1], kp[i + 2], kp[i + 3]))
                thread_list.append(thread)
                i += 1

            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()

            i += 2

    datafile = "sinc_03.xlsx"
    data = pd.read_excel(datafile, header=0)
    x = np.arange(-20.0, 20.0, 0.1)
    noise = 0.2

    # create_data(datafile, draw=True)
    researches(data)
    # one_research(data, [3.45], 300.0)

if __name__ == '__main__':
    main()
