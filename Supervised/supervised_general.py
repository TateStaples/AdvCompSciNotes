import numpy as np

class SupervisedInterface:
    def __init__(self, data, results):
        self.data = data
        self.results = results
        if isinstance(self, SupervisedInterface):
            raise Exception("Do not initialize an interface")

    def _preprocess(self):
        """
        Private function to convert data and assert it is viable
        :param data: input data
        :param results: results of data
        :return: reformatted data and results
        """
        self.data = np.array(self.data)
        seflresults = np.array(results)
        assert len(results.shape) == 1, "There should be one result per datum"
        assert len(data) == len(results), "The amount of data should equal the amount of results"
        data = np.c_[np.ones((data.shape[0], 1)), data]
        return data, results

    def __call__(self, datum):
        self.predict(datum)

    def train(self, *args):
        pass

    def predict(self, datum):
        pass
