import numpy as np
import pandas as pd

from perceptron import Perceptron


def main():
    # OR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    p_or = Perceptron(0.1, 10)
    p_or.fit(X, y)
    for i, x in enumerate(X):
        print("Input: {} | Prediction: {} | True: {}".format(x, p_or.predict(np.array([x])), y[i]))
    p_or.plot_boundary(X, y)

    # AND
    y = np.array([0, 0, 0, 1])
    p_and = Perceptron(0.1, 10)
    p_and.fit(X, y)
    for i, x in enumerate(X):
        print("Input: {} | Prediction: {} | True: {}".format(x, p_and.predict(np.array([x])), y[i]))
    p_and.plot_boundary(X, y)

    # XOR
    y = np.array([0, 1, 1, 0])
    p_xor = Perceptron(0.1, 10)
    p_xor.fit(X, y)
    for i, x in enumerate(X):
        print("Input: {} | Prediction: {} | True: {}".format(x, p_and.predict(np.array([x])), y[i]))
    p_xor.plot_boundary(X, y)

    # Iris Dataset
    X = pd.read_csv("data/IRISdata.csv", usecols=["SepalWidthCm", "PetalWidthCm"]).to_numpy()
    y = pd.read_csv("data/IRIStargets.csv", usecols=["Species"])
    y[y.Species == -1] = 0
    y = y.to_numpy()
    p = Perceptron(0.01, 100)
    p.fit(X, y)
    for i, x in enumerate(X):
        print("Input: {} | Prediction: {} | True: {}".format(x, p_and.predict(np.array([x])), y[i]))
    p.plot_boundary(X, y, start=X.min() - 2, stop=X.max() + 2, num=100, labels=False)


if __name__ == '__main__':
    main()
