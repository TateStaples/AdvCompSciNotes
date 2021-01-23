import math
import numpy as np


class KMeans:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = []

    @staticmethod
    def distance(point1, point2):
        """
        Find the euclidean distance between two points
        :param point1: nth dimensional point
        :param point2: nth dimensional point
        :return: euclidean distance between the two points
        """
        return math.sqrt(sum([(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]))

    def cluster(self, data):
        for i in range(self.k):
            self.centroids.append(data[i])
        for i in range(self.max_iterations):
            self.clusters = [[] for i in range(self.k)]

            # find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                distances = [self.distance(features, centroid) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.clusters[classification].append(features)

            previous_centroids = np.array(self.centroids)
            self.centroids = np.array([np.mean(self.clusters[i], axis=0) for i in range(self.k)])

            change = self.centroids - previous_centroids
            if change < self.tolerance:
                break

    def predict(self, datum):
        return np.argmin(self.distance(datum, centroid) for centroid in self.centroids)


if __name__ == '__main__':
    test1 = [[1, 2, 3], [1, 2, 4]]
    array = np.array(test1)
    print(np.copy(array))