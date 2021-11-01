import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        """
        Approximates a linear function that separates the samples in X into two classes

        :param X: array<n, m> comprising n samples with m features
        :param y: array<n> containing the ground truth values for each sample
        """
        # initialise the weights and bias to random values drawn from a normal distribution
        self.weights = np.random.rand(X.shape[1] + 1)

        # extend the samples with a column comprising 1s due to the bias
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        # iterate through the dataset multiple times as defined by epochs
        for epoch in range(self.epochs):
            for i, x in enumerate(X):
                # compute the activation value
                pred = x.dot(self.weights)

                # thresholding to 0 or 1 based on the activation value
                pred = 1 if pred > 0 else 0

                # update the weights based on the error and learning rate
                self.weights += self.learning_rate * (y[i] - pred) * x

    def predict(self, X):
        """
        Predicts the class for each sample (row) in X
        :param X: array<n, m> comprising n samples with m features
        :return: array<n> containing the predicted class for each sample
        """
        # extend the samples with a column comprising 1s due to the bias
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        # compute the activation value and use thresholding
        pred = X.dot(self.weights)
        pred[pred > 0] = 1
        pred[pred <= 0] = 0
        return pred

    def plot_boundary(self, X, y_true, start=-1, stop=1.5, num=20, labels=True):
        f = lambda x: (-(self.weights[2] / self.weights[1]) / (self.weights[2] / self.weights[0])) * x + (
                -self.weights[2] / self.weights[1])
        f = np.vectorize(f)
        xx = np.linspace(start, stop, num)
        fig, ax = plt.subplots()
        ax.plot(xx, f(xx))
        x = X[:, 0]
        y = X[:, 1]
        c = y_true
        ax.scatter(x=x, y=y, c=c)
        if labels:
            for i, sample in enumerate(X):
                ax.annotate("  ({}, {})".format(sample[0], sample[1]), (x[i], y[i]))
        plt.show()
