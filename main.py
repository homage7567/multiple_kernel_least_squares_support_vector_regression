import sys
import lssvm
import numpy as np
import pandas as pd
from timer import Timer
import multiprocessing as mp
from cross_validation import CV
from matplotlib import pyplot as plt


def f(x):
    return 2 * np.e ** (-0.1 * x) * np.sin(0.1 * 2 * np.pi * x)


def function_estimation(data, mutex, kp1, kp2, kp3, kp4):
    def plot_f(x, y, x_est, y_est):
        plt.figure(dpi=100, figsize=(32, 16))
        plt.plot(x, y, label="y")
        data = sorted(zip(x_est, y_est))
        x_ = [x[0] for x in data]
        y_ = [y[1] for y in data]
        plt.plot(x_, y_, label="y")
        plt.savefig(file_out + ".jpeg")

    kernel_params = [kp1, kp2, kp3, kp4]
    for r in range(10, 200, 20):
        r /= 10.0
        file_out = "Research\\" + str(kernel_params) + ", " + str(r)
        sys.stdout = open(file_out + ".txt", "w+")
        kernel_list = [lssvm.Kernel("gauss", [kernel_param]) for kernel_param in kernel_params]
        classifier = lssvm.LSSVMRegression(kernel_list, c=r)
        with Timer("CV"):
            x_est, y_est = CV.cross_val_score(data.drop("y", axis=1), data.drop("x", axis=1), classifier,
                                              segment_cnt=3)
        mse = classifier.calculate_mse(x_est, y_est, lambda arg: f(arg))
        print("MSE = " + str(mse))
        plot_f(data["x"], data["y"], x_est, y_est)

        with mutex:
            report_file = open("Research\\report.txt", "a")
            report_file.write(str(mse) + "\t" + str(r) + "\t\t\t" + str(kernel_params) + "\n")
            report_file.close()
    return


def main():
    def create_data(filename, draw=False):
        def generate_data(n_outliers=0):
            y = f(x)
            rnd = np.random.RandomState(seed=1)
            error = noise * rnd.rand(len(x))
            outliers = rnd.randint(0, len(x), n_outliers)
            error[outliers] *= 1
            return y + error

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

    def researches():
        report_file = open("Research\\report.txt", "w+")
        report_file.write("MSE\t\t\tReg_param\t\tKernel_param\n")
        report_file.close()
        files = ["test2.xlsx"]
        data = pd.read_excel(files[0], header=0)
        kp = [i / 1000.0 for i in range(1410, 1510, 5)]
        i = 0
        thread_list = []
        mutex = mp.Lock()

        with Timer("Time with threads"):
            while i < len(kp):
                thread = mp.Process(target=function_estimation,
                                    args=(data, mutex, kp[i], kp[i + 1], kp[i + 2], kp[i + 3]))
                thread_list.append(thread)
                i += 4

            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()

    noise = 0.1
    x = np.arange(0.0, 30.0, 0.08)
    # create_data("test2.xlsx", draw=True)
    researches()

if __name__ == '__main__':
    main()
