import sys
import lssvm
import numpy as np
import pandas as pd
from timer import Timer
from sklearn.svm import SVR
import multiprocessing as mp
from cross_validation import CV
from matplotlib import pyplot as plt


def f(x):
    return 10 * np.e ** (0.1 * x) * np.sin(0.6 * np.pi * x)
    # return 2 * np.e**(-0.1 * x) * np.sin(0.1 * 2 * np.pi * x)
    # return np.sinc(0.3*x)


def plot_results(x, y, y_est, file_out, dpi=100, figsize=(32, 16)):
    x_ = [x[i][0] for i in range(len(x))]
    y_cur = []
    for i in range(len(x)):
        y_cur.append(f(x[i]))
    plt.figure(dpi=dpi, figsize=figsize)
    plt.plot(x_, y, 'bo', label="Data", alpha=0.5)
    plt.plot(x_, y_cur, 'k', label="Real function")
    plt.plot(x_, y_est, "g-", label="Estimated function")
    plt.legend()
    plt.savefig(file_out + ".jpeg")


def start_estimation(X_data, Y_data, regr_estimator, file_out):
    with Timer("Cross Validation"):
        cv_score = CV.cross_val_score(X_data, Y_data, regr_estimator, f, segment_cnt=5)

    with Timer("Fit and Predict"):
        regr_estimator.fit(X_data, Y_data)
        y_est = regr_estimator.predict(X_data)

    mse = CV.calculate_mse(X_data, y_est, lambda arg: f(arg))

    print("CV score = " + str(cv_score))
    print("MSE = " + str(mse))
    plot_results(X_data, Y_data, y_est, file_out)
    return mse, cv_score


def research_LSSVR(X_data, Y_data, kernel_params, reg_param):
    file_out = "Research\\" + str(kernel_params) + ", " + str(reg_param)
    sys.stdout = open(file_out + ".txt", "w+")
    kernel_list = [lssvm.Kernel("rbf", [kernel_param]) for kernel_param in kernel_params]
    regr_estimator = lssvm.MKLSSVR(kernel_list, c=reg_param)
    mse, cv_score = start_estimation(X_data, Y_data, regr_estimator, file_out)
    betas = regr_estimator.get_betas()
    return mse, cv_score, betas


def research_SVR(X_data, Y_data, kernel_param, reg_param):
    file_out = "Research\\" + str(kernel_param[0]) + ", " + str(reg_param)
    sys.stdout = open(file_out + ".txt", "w+")
    svr_rbf = SVR(kernel='rbf', C=reg_param, gamma=kernel_param[0])
    mse, cv_score = start_estimation(X_data, Y_data, svr_rbf, file_out)
    return mse, cv_score


def research_block(research_mode, X_data, Y_data, mutex, *kernels):
    kernel_params = [k for k in kernels]
    for r in range(r_min, r_max, r_step):
        r /= r_delta
        mse, cv, betas = 0.0, 0.0, 0.0

        if research_mode == "LSSVR":
            mse, cv, betas = research_LSSVR(X_data, Y_data, kernel_params, r)
        if research_mode == "SVR":
            mse, cv = research_SVR(X_data, Y_data, kernel_params, r)

        with mutex:
            report_file = open("Research\\report.txt", "a")
            report_file.write(str(mse) + "\t" + str(cv) + "\t" + str(r) + "\t" + str(kernel_params) + "\t" +
                              str(betas) + "\n")
            report_file.close()
    return

# Параметры модедирования
r_min, r_max, r_step, r_delta = 1, 100, 10, 1.0
k_min, k_max, k_step, k_delta = 1, 10, 1, 1.0


def main():
    def data_generate(filename, draw=False):

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

        noise = 4.2
        x = np.arange(-10.0, 30.0, 0.1)
        y = generate_data(n_outliers=4)
        if draw: plot_data()
        df = pd.DataFrame({
            'x': x,
            "y": y})
        df.to_excel(filename)

    def researches(research_mode):
        report_file = open("Research\\report.txt", "w+")
        report_file.write("MSE\tCV\tReg_param\tKernel_param\tKernel_weight\n")
        report_file.close()
        kp = [i / k_delta for i in range(k_min, k_max, k_step)]
        i = 0
        thread_list = []
        mutex = mp.Lock()

        with Timer("Time with threads"):
            while i < len(kp) - 4:
                thread = mp.Process(target=research_block,
                                    args=(research_mode, X_data, Y_data, mutex, kp[i], kp[i + 1], kp[i + 2], kp[i + 3]))
                                    #args=(research_mode, X_data, Y_data, mutex, kp[i]))
                thread_list.append(thread)
                i += 2

            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()

    datafile = "Datasets\\exp_sin_10.xlsx"
    data = pd.read_excel(datafile, header=0)
    X_data = data.drop("y", axis=1).as_matrix()
    Y_data = np.array(data["y"])

    # data_generate(datafile, draw=True)
    researches("LSSVR")
    # research_LSSVR(X_data, Y_data, [0.1, 1.0, 10.0], 1.0)
    # research_SVR(X_data, Y_data, 1.0, 1.0)

if __name__ == '__main__':
    main()
