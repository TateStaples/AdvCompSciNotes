import numpy as np
from Supervised.supervised_general import SupervisedInterface


class LinearRegressor(SupervisedInterface):
    """
    Class to fit data on several types of linear regression models
    """
    def __init__(self):
        self.coefficients = None
        self.residual = None

    def _residual(self, data, results):
        """
        private function to calculate the residual of the current model
        :param data:
        :param results:
        :return:
        """
        error = results - data @ self.coefficients
        self.residual = np.linalg.norm(error)

    @staticmethod
    def _preprocess(data, results):
        """
        Private function to convert data and assert it is viable
        :param data: input data
        :param results: results of data
        :return: reformatted data and results
        """
        data = np.array(data)
        results = np.array(results)
        assert len(results.shape) == 1, "There should be one result per datum"
        assert len(data) == len(results), "The amount of data should equal the amount of results"
        data = np.c_[np.ones((data.shape[0], 1)), data]
        return data, results

    def train(self):
        pass

    def linear_regression(self, data, results):
        """
        returns a linear fit for data with no constraints
        :param data = set of numbers (all entries should be the same length)
        :param results = number representing the dependent variable
        """
        data, results = self._preprocess(data, results)

        fit = np.linalg.inv((data.T.dot(data))).dot(data.T).dot(results)
        self.coefficients = fit
        self._residual(data, results)

    def lasso_regression(self, data, results, learning_rate=0.1, iterations=100, penalty=500):
        """
        An implementation to fit data on a lasso regression algorithm
        :param data: numbers representing the independent variables
        :param results: The resulting output of the corresponding data
        :param learning_rate: How quickly new data will alter the model, default is 0.1
        :param iterations: How many steps the gradient descent takes
        :param penalty: How strongly to penalize large weights, default is 500
        """
        # https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/
        data, results = self._preprocess(data, results)
        length, dimension = data.shape
        self.coefficients = np.zeros(dimension)
        for i in range(iterations):
            prediction = self.predict(data)
            gradients = np.zeros(dimension)
            for dim in range(1, dimension):  # treat bias differently
                pen = penalty if self.coefficients[dim] > 0 else -penalty
                gradients[dim] = (-2*data.dot(results-prediction) + pen) / length
            gradients[0] = -2 * np.sum(results-prediction) / length
            self.coefficients = self.coefficients - learning_rate * gradients

    def ridge_regression(self, data, results, alpha=1):  # ridge with alpha = 1 is linear
        """
        Implementation of ridge regression
        :param data: data to fit
        :param results: the results of the data
        :param alpha: how smooth it should be - 0 means all coeff will be, 1 is normal linear regression
        """
        # https://towardsdatascience.com/how-to-code-ridge-regression-from-scratch-4b3176e5837c
        data, results = self._preprocess(data, results)

        # create weird identity matrix
        identity = np.identity(data.shape[0])
        identity[0, 0] = 0
        alpha_bias = identity * alpha

        fit = np.linalg.inv(data.T.dot(data) + alpha_bias).dot(data.T).dot(results)
        self.coefficients = fit
        self._residual(data, results)

    def elastic_net(self, data, results, learning_rate=0.1, iterations=100, penalty1=500, penalty2=1):
        """
        A combination thing of ridge and lasso
        :param data: dataset to train on
        :param results: results of dataset
        :param learning_rate: how quickly to value new updates
        :param iterations: How many gradient descents to take
        :param penalty1: L1 penalty, idk really know what that means
        :param penalty2: L2 penalty, idk really know what that means
        """
        data, results = self._preprocess(data, results)
        length, dimension = data.shape
        self.coefficients = np.zeros(dimension)
        for i in range(iterations):
            prediction = self.predict(data)
            gradients = np.zeros(dimension)
            for dim in range(1, dimension):  # treat bias differently
                pen = penalty1 if self.coefficients[dim] > 0 else -penalty1
                gradients[dim] = (-2 * data.dot(results - prediction)
                                  + pen + 2 * penalty2 * self.coefficients[dim]) / length
            gradients[0] = -2 * np.sum(results - prediction) / length
            self.coefficients = self.coefficients - learning_rate * gradients

    def predict(self, datum):
        assert self.coefficients is not None, "Please fit data before trying to predict"
        datum, _ = self._preprocess(datum, [0])
        return sum(val * coeff for val, coeff in zip(datum, self.coefficients))


if __name__ == '__main__':
    pass
