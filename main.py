import lssvm
import numpy as np


def main():
    f = lambda x: x + 4 * np.exp(-2 * x**2) / np.sqrt(2 * np.pi)
    model = lssvm.Model()
    model.read_from_excel('test_data.xlsx')
    classifier = lssvm.LSSVMRegression()

if __name__ == '__main__':
    main()
