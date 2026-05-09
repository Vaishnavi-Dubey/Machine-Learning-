"""
Principal Component Analysis (PCA) from Scratch (NumPy).
A dimensionality reduction technique that transforms data into a new coordinate system.

Math:
1. Standardize the data.
2. Compute the Covariance Matrix: Σ = (1/n) * XᵀX
3. Compute Eigenvectors and Eigenvalues: Σv = λv
4. Project data onto top 'k' eigenvectors.
"""

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # 1. Mean Centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # 2. Covariance Matrix
        # rowvar=False because columns are features
        cov = np.cov(X.T)

        # 3. Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # 4. Sort Eigenvectors by Eigenvalues (descending)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # 5. Store top 'n' components
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # Project data
        X = X - self.mean
        return np.dot(X, self.components.T)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Projected shape: {X_projected.shape}")
    print(f"Explained Variance (Components):\n{pca.components}")
