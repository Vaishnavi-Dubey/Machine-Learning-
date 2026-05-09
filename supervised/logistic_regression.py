"""
Logistic Regression from Scratch (NumPy).
Implements Binary Classification using the Sigmoid function and Log Loss.

Math:
- Sigmoid: σ(z) = 1 / (1 + e⁻ᶻ)
- Loss (Log Loss): J = -(1/n) * Σ[y log(ŷ) + (1-y) log(1-ŷ)]
"""

import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)

    model = LogisticRegressionScratch(learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)

    accuracy = np.sum(preds == y) / len(y)
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
