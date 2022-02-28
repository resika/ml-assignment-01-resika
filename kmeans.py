from cluster import cluster
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class kmeans_own(cluster):
    balanced: bool # if True, fit method uses extended kmeans

    def __init__(self, k:int=5, max_iterations:int=100, tol:float=0.001, balanced=False):
        super().__init__(k=k, max_iterations=max_iterations, tol=tol)
        self.balanced = balanced


    def fit(self, X:np.ndarray):
        self.initialize_centroids(X)
        self.class_list = np.zeros(dtype=np.int32, shape=len(X))
        for i in range(self.max_iterations):
            for i in range(self.k):
                self.classifications[i] = []

            # Classify points into clusters according to minimum distance
            for j, featureset in enumerate(X):
                distances = self.calculate_distance(featureset)
                classification = distances.index(min(distances))
                self.class_list[j] = classification
                self.classifications[classification].append(featureset)

            self._prev_centroids = dict(self.centroids)

            # Move centroids to the center of the new clusters
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            #Check whether the centroids have not moved much
            optimized = self.check_convergence()
            # scatter_plot_cluster_2d(self.classifications, self.centroids) # Debugging
            if optimized:
                data = list(self.centroids.values())
                centroid_array = np.array(data)
                return self.class_list, centroid_array

class kmeans_sk(cluster):
    def __init__(self, k:int=5, max_iterations:int=100):
        super().__init__(k=k, max_iterations=max_iterations)
        self.kmeans = KMeans(n_clusters=4, random_state=99, max_iter=self.max_iterations)

    def fit(self, X:list):
        self.kmeans.fit(X)
        return self.kmeans.labels_, self.kmeans.cluster_centers_

def scatter_plot_cluster_2d(classifications, centroids, X, title=''):
    cmap = plt.cm.get_cmap('hsv', len(centroids) + 1)
    pd_classification = pd.DataFrame(classifications)
    for classification in pd_classification[0].unique():
        g_df = X[pd_classification[0] == classification]
        plt.scatter(g_df[:,0], g_df[:,1], color=cmap(classification), s=5, alpha=0.75)
        plt.title(title)
    # Plot centroids
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1],
                    marker="o", color="k", s=50, linewidths=5)
    plt.show()

def main():
    X, cluster_assignments = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=0)

    #sklearn
    kmeans_sk_test = kmeans_sk(4, 100)
    labels, centroids = kmeans_sk_test.fit(X)
    scatter_plot_cluster_2d(labels, centroids, X)

    #own
    kmeans_own_test = kmeans_own(k=4)
    labels, centroids = kmeans_own_test.fit(X)
    scatter_plot_cluster_2d(labels, centroids, X)

if __name__ == '__main__':
    main()
