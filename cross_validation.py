import numpy as np
import pandas as pd


class CV(object):
    @staticmethod
    def cross_val_score(X, Y, classificator, segment_cnt=10):
        result = []
        size = len(X) // segment_cnt

        data = list(zip(X["x"], Y["y"]))
        np.random.shuffle(data)
        data = np.array(data)
        X_data = pd.Series([x[0] for x in data])
        Y_data = pd.Series([y[1] for y in data])

        # Первый блок
        x_test = X_data[0:size]
        x_train = X_data[size:]
        y_train = Y_data[size:]
        print("Train size: " + str(len(x_train)) + "; Test size: " + str(len(x_test)))
        print("Starting block 1!")
        classificator.fit(x_train, y_train)
        for x in classificator.predict(x_test): result.append(x)
        print("Block 1 complete!\n")

        # Внутренние блоки
        for i in range(1, segment_cnt - 1):
            x_train = X_data[:i * size]
            y_train = Y_data[:i * size]
            x_test = X_data[i * size:(i + 1) * size]
            x_train = pd.concat([x_train, X_data[(i + 1) * size:]], ignore_index=True)
            y_train = pd.concat([y_train, Y_data[(i + 1) * size:]], ignore_index=True)
            print("Train size: " + str(len(x_train)) + "; Test size: " + str(len(x_test)))
            print("Starting block " + str(i + 1) + "!")
            classificator.fit(x_train, y_train)
            for x in classificator.predict(x_test): result.append(x)
            print("Block " + str(i + 1) + " complete!\n")

        # Последний блок
        x_test = X_data[(segment_cnt - 1) * size:]
        x_train = X_data[:(segment_cnt - 1) * size]
        y_train = Y_data[:(segment_cnt - 1) * size]
        print("Starting final block")
        print("Train size: " + str(len(x_train)) + "; Test size: " + str(len(x_test)))
        classificator.fit(x_train, y_train)
        for x in classificator.predict(x_test): result.append(x)
        print("All blocks complete!\n")

        return X_data, result