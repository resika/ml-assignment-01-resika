from cluster import cluster
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import make_blobs
from random import sample

class kmeans_own(cluster):
    def __init__(self, k:int=5, max_iterations:int=100, tol:float=0.001):
        super().__init__(k=k, max_iterations=max_iterations, tol=tol)


    def fit(self, X:np.ndarray):
        self.initialize_centroids(X)
        for i in range(self.max_iterations):
            for i in range(self.k):
                self.classifications[i] = []

            # Classify points into clusters according to minimum distance
            for featureset in X:
                distances = self.calculate_distance(featureset)
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            self._prev_centroids = dict(self.centroids)

            # Move centroids to the center of the new clusters
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            #Check whether the centroids have not moved much
            optimized = self.check_convergence()
            self.scatter_plot() # Debugging
            if optimized:
                return self.classifications, self.centroids

class kmeans_sk(cluster):
    def __init__(self, k:int=5, max_iterations:int=100):
        super().__init__(k=k, max_iterations=max_iterations)
        self.kmeans = KMeans(n_clusters=4, random_state=99, max_iter=self.max_iterations)

    def fit(self, X:list):
        self.kmeans.fit(X)
        return self.kmeans.labels_, self.kmeans.cluster_centers_



def main():
    X, cluster_assignments = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=0)
    kmeans_own_test = kmeans_own(k=4)
    out = kmeans_own_test.fit(X)

if __name__ == '__main__':
    main()
