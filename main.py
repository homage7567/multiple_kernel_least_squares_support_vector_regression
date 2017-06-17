import sys
import lssvm
import numpy as np
import scipy
import pandas as pd
from timer import Timer
from sklearn.svm import SVR
import multiprocessing as mp
from cross_validation import CV
from matplotlib import pyplot as plt


def f(x):
    # return 10 * np.e ** (0.1 * x) * np.sin(0.6 * np.pi * x) # Функция для проверки rbf
    # return x**3 - 20*x**2 + 100*x - 40 # Полиномиальная функция
    # return np.sinc(0.3*x) # Категоричесий синус
    # return np.sin(x) / x # Стандартная функция для тестирования
    if x > 3.5 and x < 5.5:
        return np.e**(-(x - 4.7)**2)/0.3
    if x > 9.8 and x < 13.2:
        return 0.06*x**2 + np.e**(-(x - 11.5)**2)/0.3
    else:
        return 0.06*x**2


def plot_results(X_data, Y_data, y_est, X_test, file_out, dpi=200, figsize=(10, 6)):
    x_cur = [X_data[i][0] for i in range(len(X_data))]
    x_est = [X_test[i][0] for i in range(len(X_test))]
    y_cur = []
    for i in range(len(x_est)):
        y_cur.append(f(x_est[i]))

    plt.figure(dpi=dpi, figsize=figsize)
    plt.grid(True)
    plt.xlabel("X (attrubute)", fontsize=14)
    plt.ylabel("Y (response)", fontsize=14)
    plt.plot(x_cur, Y_data, 'bo', label="Data", alpha=0.5)
    plt.plot(x_est, y_cur, 'r--', label="Real function", alpha=0.7)
    plt.plot(x_est, y_est, "k-", label="Estimated function")
    plt.legend(prop={'size': 14})
    plt.savefig(file_out + ".jpeg")


def start_estimation(X_data, Y_data, X_test, regr_estimator, file_out):
    with Timer("Cross Validation"):
        cv_score = CV.cross_val_score(X_data, Y_data, regr_estimator, f, segment_cnt=5)

    with Timer("Fit and Predict"):
        regr_estimator.fit(X_data, Y_data)
        y_est = regr_estimator.predict(X_test)

    mse = CV.calculate_mse(X_test, y_est, lambda arg: f(arg))

    print("CV score = " + str(cv_score))
    print("MSE = " + str(mse))
    plot_results(X_data, Y_data, y_est, X_test, file_out)
    return mse, cv_score


def research_LSSVR(X_data, Y_data, X_test, kernel_params, reg_param):
    file_out = "Research\\" + str(kernel_params) + ", " + str(reg_param)
    sys.stdout = open(file_out + ".txt", "w+")

    kernel_list = []
    if len(kernel_params) > 1:
        tmp = int(len(kernel_params) / 2)
        for i in range(tmp):
            kernel_list.append(lssvm.Kernel("rbf", [kernel_params[i]]))
        for i in range(tmp, len(kernel_params)):
            kernel_list.append(lssvm.Kernel("poly", [kernel_params[i], 2]))
    else:
        kernel_list = [lssvm.Kernel("rbf", [kernel_params[0]])]
    regr_estimator = lssvm.MKLSSVR(kernel_list, c=reg_param)
    mse, cv_score = start_estimation(X_data, Y_data, X_test, regr_estimator, file_out)
    betas = regr_estimator.get_betas()
    return mse, cv_score, betas


def research_SVR(X_data, Y_data, X_test,kernel_param, reg_param):
    file_out = "Research\\" + str(kernel_param[0]) + ", " + str(reg_param)
    sys.stdout = open(file_out + ".txt", "w+")
    svr_rbf = SVR(kernel='poly', degree=2, C=reg_param, gamma=kernel_param[0])
    mse, cv_score = start_estimation(X_data, Y_data, X_test, svr_rbf, file_out)
    return mse, cv_score


def research_block(research_mode, X_data, Y_data, X_test, mutex, *kernels):
    kernel_params = [k for k in kernels]
    for r in reg:
        mse, cv, betas = 0.0, 0.0, 0.0

        if research_mode == "LSSVR":
            mse, cv, betas = research_LSSVR(X_data, Y_data, X_test, kernel_params, r)
        if research_mode == "SVR":
            mse, cv = research_SVR(X_data, Y_data, X_test, kernel_params, r)

        with mutex:
            report_file = open("Research\\report.txt", "a")

            str_kp = ""
            for kp in kernel_params:
                str_kp += '{:.3f} '.format(kp)
            str_beta = ""
            for beta in betas:
                str_beta += '{:.3f} '.format(beta)

            report_file.write('{:.3f}'.format(mse[0]) + '\t{:.3f}'.format(cv[0]) + "\t" + str(r) + "\t" +
                              str_kp + "\t" + str_beta + "\n")
            report_file.close()
    return

# Параметры модедирования
# kp = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
kp = [1, 2, 3, 4, 5, 6]
# kp = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
# kp = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
# kp_mk = [3.0, 10.0]
# reg = [0.0001, 0.001, 0.1, 1.0, 2.0, 4.0, 6.0, 10.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 1000.0]
reg = [0.0001, 0.001, 0.1, 1.0, 2.0, 4.0, 6.0, 10.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 1000.0]
# kp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# reg = [1, 31, 61, 91, 121, 151, 181, 211, 241, 271]


def main():
    def data_generate(directory, function_name, x, noize_percent=0.75):

        def create_noize():
            u_mid, w = 0.0, 0.0
            for y_elem in y:
                u_mid += y_elem
            u_mid /= len(x)
            for i in range(len(x)):
                w = (y[i] - u_mid) ** 2
            gamma = noize_percent * w / (len(x) - 1)
            errors = scipy.random.normal(loc=0.0, scale=np.sqrt(gamma), size=len(x))
            return errors

        def save_figure(dpi=200, figsize=(10, 6)):
            plt.figure(dpi=dpi, figsize=figsize)
            plt.grid(True)
            plt.xlabel("X (attrubute)", fontsize=14)
            plt.ylabel("Y (response)", fontsize=14)
            y_cur = [f(i) for i in x]
            plt.plot(x, y_cur, 'r--', label="Real Function")
            plt.plot(x, y, 'bo', label='Data', alpha=0.5)
            plt.legend(loc='upper center', prop={'size': 14})
            plt.savefig(filename + ".jpeg")

        y = []
        for i in x:
            y.append(f(i))
        y += create_noize()
        filename = directory + function_name + "_" + str(noize_percent) + "_" + str(len(x))
        print("Complete! Number of observations: " + str(len(x)))
        save_figure()
        df = pd.DataFrame({
            'x': x,
            "y": y})

        df.to_excel(filename + ".xlsx")

    def researches(research_mode, mk_mode=True):
        report_file = open("Research\\report.txt", "w+")
        report_file.write("MSE\tCV\tReg_param\tKernel_param\tKernel_weight\n")
        report_file.close()
        i = 0
        thread_list = []
        mutex = mp.Lock()

        with Timer("Time with threads"):
            if mk_mode:
                # while i < len(kp) - 3:
                #     thread = mp.Process(target=research_block, args=(research_mode, X_data, Y_data, X_test,
                #                                                      mutex, kp[i], kp[i + 1], kp[i + 2], kp[i + 3]))
                #     i += 2
                #     thread_list.append(thread)
                thread = mp.Process(target=research_block, args=(research_mode, X_data, Y_data, X_test, mutex, kp[i],
                                                                 kp[i + 1], kp[i + 2], kp[i + 3], kp[i + 4], kp[i + 5]))
                thread_list.append(thread)

            else:
                while i < len(kp):
                    thread = mp.Process(target=research_block, args=(research_mode, X_data, Y_data, X_test, mutex,
                                                                     kp[i]))
                    thread_list.append(thread)
                    i += 1

            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()

    # Исследования
    datafile = "Datasets\\wtf\\wtf_0.1_30.xlsx"
    data = pd.read_excel(datafile, header=0)
    X_data = data.drop("y", axis=1).as_matrix()
    X_test = pd.read_excel("Datasets\\wtf\\test.xlsx", header=0).as_matrix()
    Y_data = np.array(data["y"])
    researches("LSSVR", mk_mode=True)

    # plt.figure(dpi=200, figsize=(10, 6))
    # plt.grid(True)
    # plt.xlabel("X (attrubute)", fontsize=14)
    # plt.ylabel("Y (response)", fontsize=14)
    # x_cur = [X_data[i][0] for i in range(len(X_data))]
    # x_est = [X_test[i][0] for i in range(len(X_test))]
    # y_cur = []
    # for i in range(len(x_est)):
    #     y_cur.append(f(x_est[i]))
    # plt.plot(x_cur, Y_data, 'bo', label="Data", alpha=0.5)
    # plt.plot(x_est, y_cur, 'r--', label="Real function", alpha=0.7)
    # # plt.plot(x_est, y_est, "k-", label="Estimated function")
    # plt.legend(prop={'size': 14})
    # plt.savefig("dataset.jpeg")

    # Генерация наборов данных
    # for i in [0.1]:
    #     for j in [0.5]:
    #         data_generate("Datasets\\wtf\\", "wtf", np.arange(-15.0, 15.0, j), i)


if __name__ == '__main__':
    main()
