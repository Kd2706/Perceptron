import numpy as np

class MyPerceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
       self.lr = learning_rate
       self.epochs = n_iterations
       self.w = None   # weights
       self.b = None   #bias

    def fit(self, x, y):
        self.w = np.ones(x.shape[1])
        self.b = 0

        for i in range(self.epochs):
            y_hat = self.activation(np.dot(self.w, x.T) + self.b)

            # dL = -np.mean(y * np.log(y_hat) - (1-y) * np.log(1-y_hat))
            # dL is the cost or binary cross entropy error

            dw = - np.dot((y - y_hat), x) / x.shape[0]  # derivation of loss  with respect to w : dL/dw
            db = -np.mean(y - y_hat)   # derivation of loss with respect to b: dL/db

            self.w = self.w - (self.lr * dw)   # updating w and b after each gradient descent on loss
            self.b = self.b - (self.lr * db)

        print(self.w)
        print(self.b)

    def activation(self, z):
        return 1/(1 + np.exp(-z))   # using sigmoid as the activation function to generate y_hat

    def predict(self, x):
        y_pred = []
        for i in range(x.shape[0]):

            # convert predicted y to binary type classification of 1 or 0
            conversion = (self.activation(np.dot(self.w, x[i].T) + self.b) > 0.5).astype(int)
            y_pred.append(conversion)
        return np.array(y_pred)  # entire prediction of y on test data

