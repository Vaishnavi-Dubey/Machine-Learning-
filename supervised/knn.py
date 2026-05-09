"""
K-Nearest Neighbors (KNN) from Scratch (NumPy).
A non-parametric, lazy learning algorithm for classification.

Concept: Finds the 'k' most similar training examples (nearest neighbors) 
and performs a majority vote to predict the class.
"""

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # 1. Compute distances between x and all examples in training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 2. Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target

    clf = KNN(k=5)
    clf.fit(X, y)
    preds = clf.predict(X[:10]) # Predict first 10
    
    print(f"KNN Predictions for first 10: {preds}")
    print(f"Actual labels for first 10:   {y[:10]}")
