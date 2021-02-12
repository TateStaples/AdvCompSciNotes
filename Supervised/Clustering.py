import math
import numpy as np
from Supervised.supervised_general import SupervisedInterface


class KNearestNeighbors(SupervisedInterface):
    # https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

    def __init__(self, data, results, k, use="classification"):
        self.data, self.results = data, results

    def train(self, k, use="classification"):
        self.k, self.use = k, use

    @staticmethod
    def distance(point1, point2):
        """
        Find the euclidean distance between two points
        :param point1: nth dimensional point
        :param point2: nth dimensional point
        :return: euclidean distance between the two points
        """
        return math.sqrt(sum([(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]))

    @staticmethod
    def weighted_average(data):
        total_weight = np.sum(data[:, -1])
        total_val = sum(weight * val for weight, val in data)
        return total_val/total_weight

    def predict(self, point):
        distances = [self.distance(datum, point) for datum in self.data]
        combined = list(zip(distances, self.results))
        sorted_data = np.array(sorted(combined, key=lambda tup: tup[0]))
        if self.use == "classification":
            results = np.array(sorted_data)[:, 1]
            return np.mode(results)
        else:  # this should be regression
            return self.weighted_average(sorted_data)


class NaiveBayes:
    # https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
    def __init__(self, data, classes):
        self.data = data
        self.results = classes

    def train(self):
        self.classes = self._create_classes()

    def gaussian_probability(self, datum, class_index):
        _, mean, std = self.classes[class_index][0]
        return np.mean([(1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((x-mean)**2 / (2 * std**2))) for x in datum])

    def _create_classes(self):
        groups = dict()
        for datum, result in zip(self.data, self.results):
            if result in groups:
                groups[result].append(datum)
            else:
                groups[result] = [datum]
        classes = []
        for classification in groups:
            class_data = groups[classification]
            means = np.mean(class_data, axis=0)  # mean for each axis
            stds = np.std(class_data, axis=0)  # standard deviation for each axis
            summary = (classification, means, stds)
            classes.append((summary, class_data))
        return classes

    def predict(self, datum):
        best = np.argmax([self.gaussian_probability(datum, i) for i in range(len(self.classes))])
        classification, mean, std = self.classes[best[0]]
        return classification
