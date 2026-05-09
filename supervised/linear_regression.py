"""
Linear Regression from Scratch (NumPy).
Implements Ordinary Least Squares (OLS) using Gradient Descent.

Math:
- Prediction: ŷ = Xw + b
- Loss (MSE): J(w,b) = (1/2n) * Σ(ŷ - y)²
- Gradients: ∂J/∂w = (1/n) * Xᵀ(ŷ - y), ∂J/∂b = (1/n) * Σ(ŷ - y)
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # Toy Data
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100)

    model = LinearRegressionScratch(learning_rate=0.1)
    model.fit(X, y)
    predictions = model.predict(X)

    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")

    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(X, predictions, color='red', label='Prediction')
    plt.title('Linear Regression from Scratch')
    plt.legend()
    print("Demo complete. Coefficients trained successfully.")
