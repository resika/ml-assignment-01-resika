import numpy as np


class cluster:

    def __init__(self, k: int = 5, max_iterations: int = 100, tol: float = 0.001):
        self.k = k  # target number of cluster centroids
        self.max_iterations = max_iterations  # maximum number of times to execute the convergence attempt
        self.tol = tol  # Tolerance
        self.centroids = {}  # Centroids
        self.classifications = {}  # Classifications
        self._prev_centroids = {}

    def fit(self, X: list):
        pass

    def initialize_centroids(self, X):
        # Calculating initial centroid
        index = np.random.randint(len(X), size=self.k)
        for i in range(self.k):
            self.centroids[i] = X[index[i]]

    def calculate_distance(self, point: np.ndarray):
        distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
        return distances

    def check_convergence(self):
        optimized = True
        for c in self.centroids:
            original_centroid = self._prev_centroids[c]
            current_centroid = self.centroids[c]
            if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                optimized = False
        return optimized



