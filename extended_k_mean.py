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
        self.ideal_cluster_size = None



    def fit(self, X:np.ndarray):
        self.initialize_centroids(X)
        self.labels = np.zeros(dtype=np.int32, shape=len(X)) # Labels
        if self.balanced:
            self.ideal_cluster_size = self.calculate_desired_cluster_size(len(X))

        for i in range(self.max_iterations):
            for j in range(self.k):
                self.classifications[j] = []

            # Classify points into clusters according to minimum distance
            for j, featureset in enumerate(X):
                distances = self.calculate_distance(featureset)
                # classification = distances.index(sorted(distances)[0])
                # classfication2 = distances.index(sorted(distances)[1])
                # self.labels[j] = classification
                # self.classifications[classification].append(featureset)
                self.assign_to_clusters(distances, featureset, currentpoint_index=j)

            self._prev_centroids = dict(self.centroids)

            # Move centroids to the center of the new clusters
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            #Check whether the centroids have not moved much
            optimized = self.check_convergence()
            # scatter_plot_cluster_2d(self.classifications, self.centroids) # Debugging
            data = list(self.centroids.values())
            centroid_array = np.array(data)
            if optimized:
                return self.labels, centroid_array

            # self.print_cluster_size(i)
            # scatter_plot_cluster_2d(self.labels, centroid_array, X, title=f'Iteration: {i}')


    def calculate_desired_cluster_size(self, n:int):
        """
        Calculate desired cluster size
        :param n: Number of data points
        :return: Number of points per cluster
        """
        return n/self.k

    def assign_to_clusters(self, distances, featureset, currentpoint_index):
        """
        Assigns current point considered to clusters
        :param distances:
        :param featureset:
        :param currentpoint_index:
        :return:
        """
        if self.balanced:
            for i in range(self.k):
                label = distances.index(sorted(distances)[i])
                if (len(self.classifications[label])>self.ideal_cluster_size):
                    continue
                else:
                    self.labels[currentpoint_index] = label
                    self.classifications[label].append(featureset)
                    break
        else:
            label = distances.index(sorted(distances)[0])
            self.labels[currentpoint_index] = label
            self.classifications[label].append(featureset)

    def print_cluster_size(self, iteration_number):
        print(f"Iteration: {iteration_number}")
        for i in range(self.k):
            print(f"{i}     {len(self.classifications[i])}")

class kmeans_sk(cluster):
    def __init__(self, k:int=5, max_iterations:int=100):
        super().__init__(k=k, max_iterations=max_iterations)
        self.kmeans = KMeans(n_clusters=5, random_state=99, max_iter=self.max_iterations)

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
    k=5
    X, cluster_assignments = make_blobs(n_samples=200, centers=k, cluster_std=0.50, random_state=0)

    # #sklearn
    # kmeans_sk_test = kmeans_sk(k, 100)
    # labels, centroids = kmeans_sk_test.fit(X)
    # scatter_plot_cluster_2d(labels, centroids, X)
    #
    # #own
    # print("\nOwn k-means")
    # kmeans_own_test = kmeans_own(k=k)
    # labels, centroids = kmeans_own_test.fit(X)
    # scatter_plot_cluster_2d(labels, centroids, X,  title="Own")

    print("\nOwn Balanced k-means")
    # own balanced
    kmeans_own_test = kmeans_own(k=k, balanced=True)
    labels, centroids = kmeans_own_test.fit(X)
    scatter_plot_cluster_2d(labels, centroids, X, title="Own Balanced")

if __name__ == '__main__':
    main()
