import numpy as np


class LogisticRegression:

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def h(self, theta, X, boundary):
        """
        Makes a prediction based on the parameters of a model.
        One of the two dimensions of each matrix must be equal for a
        prediction to be made.
        :param theta:
        :param X:
        :return:
        """
        if not 0 < boundary < 1:
            print('Boundary should be between zero and 1')
            return

        r, c = theta.shape
        if r > c:
            theta = theta.transpose()

        r, c = X.shape
        if c < r:
            X = X.transpose()

        return np.greater(self.sigmoid(theta * X), boundary)

    def temporary_tests(self):
        theta = np.matrix('1 1 1').transpose()
        X = np.matrix('1 2 3; 1 1 1; 1 1 1; 1 1 1')
        boundary = 0.999

        print(self.h(theta, X, boundary))
        print(self.h(theta.transpose(), X, boundary))
        print(self.h(theta, X.transpose(), boundary))
        print(self.h(theta.transpose(), X.transpose(), boundary))

        self.h(theta.transpose(), X.transpose(), -1)
        self.h(theta.transpose(), X.transpose(), 0)
        self.h(theta.transpose(), X.transpose(), 0.5)
        self.h(theta.transpose(), X.transpose(), 1)
        self.h(theta.transpose(), X.transpose(), 1.5)


if __name__ == '__main__':
    model = LogisticRegression()
    model.tests()
