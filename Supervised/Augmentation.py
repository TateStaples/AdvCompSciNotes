from Supervised.supervised_general import SupervisedInterface
import numpy as np


class Bagging(SupervisedInterface):  # this is like random forest
    def train(self, amount: int, sample_size: int, _class, args: tuple = ()):
        self.models = list()
        for _ in range(amount):
            sample = np.random

    def predict(self, datum):
        return np.mean([m.predict(datum) for m in self.models])


class GradientBoost(SupervisedInterface):
    def train(self, amount: int, _class, args: tuple = ()):
        """
        Train a series of models in a way that they create one single better model
        :param amount: how many models to include in the series - too many can cause over-fitting
        :param _class: the type of model that is to be created, Should be a sub-class of SupervisedInterface.
                        This only works for certain types of algorithms like trees and NN
        :param args: the arguments for how your _class should be trained
        :return:
        """
        self.models = list()
        previous_predictions = np.zeros(self.data.shape[0])
        for _ in range(amount):
            residual = self.results - previous_predictions  # what is the error of the current system
            new_model = _class(self.data, residual)  # train the next model on the error of the previous ones
            predictions = np.array(new_model.predict(datum) for datum in self.data)
            previous_predictions += predictions
            self.models.append(new_model)

    def predict(self, datum):
        return sum(m.predict(datum) for m in self.models)