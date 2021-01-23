import numpy as np


class Tree:
    # https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
    def __init__(self, data, max_depth, metric="gini"):
        """
        A decision regression tree implementation
        :param data: a 2-d list how data with the results at the end of each row
        :param max_depth: how large the tree can grow, larger means more complicated but more prone to overfittting
        :param metric: How to evaluate the quality of the tree ("gini" or "entropy")
        """
        self.max_depth = max_depth
        self.metric = metric
        self.data = np.array(data) if type(data) == list else data
        self.classes = self._classes()
        self.leaf = False
        self.depth = 0
        self.left = self.right = self.question = None

    def _create(self, data, depth):
        """
        private function to create trees
        :param data: the section of data this branch of the tree will contain
        :param depth: how far down this tree is
        :return: the created Tree
        """
        b = Tree(data, self.max_depth, self.metric)
        b.depth = depth
        return b

    def _classes(self):
        """ Finds what classes are a part of this trees dataset """
        return set(self.data[:, -1])  # get the back column which should be the labels

    @property
    def gini(self):
        """ the gini index of this Tree - odds type(random.choice(group)) == type(random.choice(group) """
        if self.left is not None:
            return self.data  # data is replaced after new nodes created
        # count all samples at split point
        n_instances = len(self.data)
        # sum weighted Gini index for each group
        gini = 0.0
        for group in self._test_split():
            size = len(group)
            if size == 0:
                continue  # avoid divide by zero
            score = 0
            # score the group based on the score for each class
            for class_val in self.classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1 - score) * (size / n_instances)
            del group
        return gini

    @property
    def entropy(self):  # todo: check this, I got it from a random comment
        """ entropy of tree - how mixed together it is """
        if self.left is not None:
            return self.data  # data is replaced after new nodes created
        length = len(self.data)
        e = 0
        for val in self.classes:
            count = 0
            for item in self.data:
                if item[2] == val:
                    count += 1
            frac = count / length
            # if frac =0, there is a perfect split, and e=0. don't add anything to
            if frac == 0:
                continue
            e += -frac * np.log(frac)
        return e

    def query(self, datum):
        """
        Check a trained tree if the datum fits that tree criteria
        :param datum: a piece of data to check
        :return: bool representing if it passes
        """
        assert self.question is not None, "You cannot query a branch before it has its criteria set"
        feature, test_value = self.question
        return datum[feature] > test_value

    def get_branch(self, datum):
        """
        Return what sub-tree the datum is a part of
        :param datum: a piece of data to check
        :return: another Tree containing the datum
        """
        if self.leaf:
            return self
        if self.query(datum):
            return self.right
        return self.left

    def _test_split(self):
        """
        Private function that splits the Trees data based on the current question
        :return: two lists based on if the data fits the question
        """
        l, r = list(), list()
        for datum in self.data:
            if self.query(datum):
                r.append(datum)
            else:
                l.append(datum)
        return l, r

    def split(self):
        """
        Takes a tree and splits it into 2 sub-trees
        :return: None
        """
        assert self.right is None and self.left is None, "Please do not re-split a Tree :("
        if self.depth >= self.max_depth:
            self.leaf = True
            return
        best_score = best_question = 999 if self.metric == "gini" else 0
        n_features = self.data.shape[1] - 1  # last one

        for feature in range(n_features):
            for row in self.data:
                self.question = feature, row[feature]
                score = self.gini
                if (self.metric == "gini" and score < best_score) or (self.metric == "entropy" and score < best_score):
                    best_score, best_question = score, self.question

        # check the split is valid - would be invalid if at max
        if best_score == 999 or best_score == 0:
            self.leaf = True
            return

        self.question = best_question
        l, r = self._test_split()
        self.left, self.right = self._create(l, self.depth+1), self._create(r, self.depth+1)
        self.data = self.gini if self.metric == "gini" else self.entropy

    def predict(self, datum):
        """
        Use the tree to predict a value
        :param datum: the piece of data to predict
        :return: the average value of teh final branch
        """
        branch = self
        while not branch.leaf:
            branch = branch.get_branch(datum)
        return np.average(branch.data[:, -1])


class RandomForest:
    # https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249
    def __init__(self, data, forest_size, sample_size, depth=5):
        """
        A group of trees that get averaged out to get a better model
        :param data: Data to fit, results should be at the end of each row
        :param forest_size: how many trees to make
        :param sample_size: how large a sample of data does each tree get
        :param depth: how large each tree can get
        """
        self.trees = list()
        for i in range(forest_size):
            sample_data = np.random.permutation(data)[:sample_size]
            self.trees.append(Tree(sample_data, depth))

    def predict(self, datum):
        """
        Use the forest to predict the result
        :param datum: datum to use for prediction
        :return: predicted result
        """
        return np.mean([tree.predict(datum) for tree in self.trees], axis=0)


class BoostedTrees:
    #  https://www.machinelearningplus.com/machine-learning/gradient-boosting/
    def __init__(self, data: np.ndarray, depth: int, num_trees: int):
        """
        A series of small tree each trained on the previous tree's error
        :param data: dataset to train on, results should be at the end of each row
        :param depth: how large each tree can get
        :param num_trees: how long the series of trees should get, too many can overfit
        """
        self.trees = list()
        previous_predictions = np.zeros(data.shape[0])
        actual_results = data[:, -1]
        for i in range(num_trees):
            residual = actual_results - previous_predictions
            data[:, -1] = residual
            t = Tree(data, depth)
            predictions = np.array([t.predict(datum[:-1]) for datum in data])
            previous_predictions += predictions
            self.trees.append(t)

    def predict(self, datum):
        """
        Use the trees to predict the result
        :param datum: datum to use for prediction
        :return: predicted result
        """
        return sum(t.predict(datum) for t in self.trees)
