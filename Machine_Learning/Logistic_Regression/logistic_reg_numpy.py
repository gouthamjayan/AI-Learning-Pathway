import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = 0.0

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))

    def fit(self, X, y):
        N,D = X.shape
        self.W = np.zeros((D,1))
        self.b = 0.0
        y = y.reshape(-1,1)

        for epoch in range(self.epochs):
            Z = np.dot(X,self.W) + self.b

            y_hat = self.sigmoid(Z)

            error = y_hat - y

            dW = (1/N) * np.dot(X.T,error)
            db = (1/N) * np.sum(error)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict_proba(self,X):
        Z = np.dot(X,self.W) + self.b
        return self.sigmoid(Z)

    def predict(self,X,threshold = 0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
