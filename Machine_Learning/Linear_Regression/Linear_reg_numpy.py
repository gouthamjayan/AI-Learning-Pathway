import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = 0.0

    def fit(self, X,y):
        N,D = X.shape
        self.W = np.zeros((D,1))
        y = y.reshape(-1,1)

        for epoch in range(self.epochs):
            y_hat = np.dot(X,self.W) + self.b
            error = y_hat - y
            dW = (2/N) * np.dot(X.T, error)
            db = (2/N) * np.sum(error)
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict_line(self,X):
        return np.dot(X,self.W) + self.b

#Generate Synthetic Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3.5 * X[:, 0] + 7.0 + np.random.randn(100) * 0.1

model = LinearRegression(learning_rate = 0.05, epochs= 1000)
model.fit(X,y)


