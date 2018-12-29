import sys
import mk_ls_svr as lssvm
import numpy as np
import scipy
import pandas as pd
from timer import Timer
from sklearn.svm import SVR
import multiprocessing as mp
from cross_validation import CV
from matplotlib import pyplot as plt


def f(x):
    return x**2
    # return 10 * np.e ** (0.1 * x) * np.sin(0.6 * np.pi * x) # Функция для проверки rbf
    # return x**3 - 20*x**2 + 100*x - 40  # Полиномиальная функция
    # return np.sinc(0.3*x) # Категоричесий синус
    # return np.sin(x) / x # Стандартная функция для тестирования


def plot_results(x, y, y_est, file_out, dpi=200, figsize=(10, 6)):
    x_ = [x[i][0] for i in range(len(x))]
    y_cur = []
    for i in range(len(x)):
        y_cur.append(f(x[i]))
    plt.figure(dpi=dpi, figsize=figsize)
    plt.grid(True)
    plt.xlabel("X (attrubute)", fontsize=14)
    plt.ylabel("Y (response)", fontsize=14)
    plt.plot(x_, y, 'bo', label="Data", alpha=0.5)
    plt.plot(x_, y_cur, 'r--', label="Real function")
    plt.plot(x_, y_est, "k-", label="Estimated function")
    plt.legend(prop={'size': 14})
    plt.savefig(file_out + ".jpeg")


def start_estimation(X_data, Y_data, regr_estimator, file_out):
    with Timer("Cross Validation"):
        cv_score = CV.cross_val_score(
            X_data, Y_data, regr_estimator, f, segment_cnt=5)

    with Timer("Fit and Predict"):
        regr_estimator.fit(X_data, Y_data)
        y_est = regr_estimator.predict(X_data)

    mse = CV.calculate_mse(X_data, y_est, lambda arg: f(arg))

    print("CV score = " + str(cv_score))
    print("MSE = " + str(mse))
    plot_results(X_data, Y_data, y_est, file_out)
    return mse, cv_score


def research_LSSVR(X_data, Y_data, kernel_params, reg_param, c_one, c_two):
    file_out = "Research\\" + str(kernel_params) + ", " + str(reg_param) + ", " + str(c_one) + ", " + str(c_two)
    sys.stdout = open(file_out + ".txt", "w+")

    kernel_list = []
    if len(kernel_params) > 1:
        tmp = int(len(kernel_params) / 2)
        for i in range(0, tmp):
            kernel_list.append(lssvm.Kernel("rbf", [kernel_params[i]]))
        for i in range(tmp, len(kernel_params)):
            kernel_list.append(lssvm.Kernel("poly", [kernel_params[i], 1]))
    else:
        kernel_list = [lssvm.Kernel("rbf", [kernel_param])
                       for kernel_param in kernel_params]
    regr_estimator = lssvm.MKLSSVR(kernel_list, c=reg_param, c_one=c_one, c_two=c_two)
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
    for r in reg:
        for c_one in c_one_r:
            for c_two in c_two_r:
                mse, cv, betas = 0.0, 0.0, 0.0

                if research_mode == "LSSVR":
                    mse, cv, betas = research_LSSVR(X_data, Y_data, kernel_params, r, c_one, c_two)
                if research_mode == "SVR":
                    mse, cv = research_SVR(X_data, Y_data, kernel_params, r)

                with mutex:
                    report_file = open("Research\\report.txt", "a")

                    str_kp = ""
                    for kp in kernel_params:
                        str_kp += '{:.3f} '.format(kp)
                    str_beta = ""
                    for beta in betas:
                        str_beta += '{:.3f} '.format(beta)

                    report_file.write('{:.3f}'.format(mse[0]) + '\t{:.3f}'.format(cv[0]) + '\t{:.3f}'.format(r) + "\t" +
                                    str_kp + "\t" + str_beta + "\n")
                    report_file.close()
    return


# Параметры модедирования
# kp = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
kp = [3.0, 4.0, 5.0, 8.0]
reg = [2.0, 4.0, 6.0, 10.0]

c_one_r = [1.0, 2.0, 3.0]
c_two_r = [1.0, 2.0, 3.0]


def main():
    def data_generate(directory, function_name, x, noize_percent=0.25, emissions=0):

        def create_noize():
            u_mid, w = 0.0, 0.0
            for y_elem in y:
                u_mid += y_elem
            u_mid /= len(x)
            for i in range(len(x)):
                w = (y[i] - u_mid) ** 2
            gamma = 0.0
            gamma = noize_percent * w / (len(x) - 1)
            
            errors = scipy.random.normal(
                loc=0.0, scale=np.sqrt(gamma), size=len(x))

            emiss_idxs = []
            for i in range(emissions):
                emiss_idx = np.random.randint(0, len(errors))
                while emiss_idx in emiss_idxs:
                    emiss_idx = np.random.randint(0, len(errors))

                emiss_idxs.append(emiss_idx)
                errors[emiss_idx] *= np.random.randint(3, 5)

            return errors

        def save_figure(dpi=200, figsize=(10, 6)):
            plt.figure(dpi=dpi, figsize=figsize)
            plt.grid(True)
            plt.xlabel("X (attrubute)", fontsize=14)
            plt.ylabel("Y (response)", fontsize=14)
            y_cur = [f(i) for i in x]
            plt.plot(x, y_cur, 'r--', label="Real Function")
            plt.plot(x, y, 'bo', label='Data', alpha=0.5)
            plt.legend(prop={'size': 14})
            plt.savefig(filename + ".jpeg")

        y = f(x)
        y += create_noize()
        filename = directory + function_name + "_" + \
            str(noize_percent) + "_" + str(len(x))
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
                    # thread = mp.Process(target=research_block, args=(research_mode, X_data, Y_data, mutex, kp[i],
                    #                                                  kp[i + 1], kp[i + 2], kp[i + 3]))

                    # i += 1
                # thread = mp.Process(target=research_block, args=(research_mode, X_data, Y_data, mutex, kp_mk[i],
                #                                                  kp_mk[i + 1], kp_mk[i +
                #                                                                      2], kp_mk[i + 3],
                #                                                  kp_mk[i +
                #                                                        4], kp_mk[i],
                #                                                  kp_mk[i + 1], kp_mk[i +
                #                                                                      2], kp_mk[i + 3],
                #                                                  kp_mk[i + 4]))
                # thread_list.append(thread)
                pass
            else:
                while i < len(kp):
                    thread = mp.Process(target=research_block, args=(
                        research_mode, X_data, Y_data, mutex, kp[i]))
                    thread_list.append(thread)
                    i += 1

            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()

    # # Исследования
    datafile = "Datasets\\x2_2\\x2_2_0.4_80.xlsx"
    data = pd.read_excel(datafile, header=0)
    X_data = data.drop("y", axis=1).as_matrix()
    Y_data = np.array(data["y"])
    researches("LSSVR", mk_mode=False)

    # Генерация наборов данных
    # for i in [0.4, 0.8]:
    #     for j in [1.6, 0.8, 0.4, 0.2, 0.1]:
    #         data_generate("Datasets\\x2_2\\", "x2_2", np.arange(-8.0, 8.0, j), i, emissions=8)


if __name__ == '__main__':
    main()
