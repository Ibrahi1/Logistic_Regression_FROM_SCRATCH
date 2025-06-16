import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=500):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weighs= np.zeros(n_features)
        self.bias=0
        
        # gradiant discent
        for _ in range(self.n_iters):
            linear_model  = np.dot(X, self.weighs) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            self.weighs -=self.lr*dw
            self.bias -=self.lr*db

    def predict(self, X):
        linear_model  = np.dot(X, self.weighs) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
def accuracy(y_true,  y_pred):
    acc = np.sum(y_true==y_pred) / len(y_true)
    return acc