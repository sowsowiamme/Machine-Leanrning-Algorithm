import numpy as np

class BasicSVM:

    def __init__(self, learning_rate=0.01, n_iters = 100, lambda_param=0.01):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters


    def fit(self, X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.b = 0

        # 确保标签是±1
        y_ = np.where(y <= 0, -1, 1)

        for iter in range(self.n_iters):
            for idx,x_i in X:
                condition = (np.dot(self.weights, x_i)+ self.bias)*y_[idx] >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    # 错误分类，更新w和b
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * (-y_[idx])


    def predict(self, X):
        linear_output = self._decision_fuction(X)
        return np.sign(linear_output)

    def _decision_fuction(self, X):
        y = np.dot(self.weights, X) + self.bias
        return y