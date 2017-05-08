"""
this module contains the RandomForest class
"""

import numpy as np
from decision_tree import DecisionTree

class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """
    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.ratio_per_tree = ratio_per_tree
        self.trees = []

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """

        if not isinstance(X, list):
            X = X.tolist()
        if not isinstance(Y, list):
            Y = Y.tolist()

        N = len(X)

        for _ in range(self.num_trees):
            batch_size = int(N*self.ratio_per_tree)
            idx = np.random.choice(N, batch_size, replace=True)
            trainX = [X[i] for i in idx]
            trainY = [Y[i] for i in idx]
            tree = DecisionTree(self.max_tree_depth)
            tree.fit(trainX, trainY)
            self.trees.append(tree)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """

        Ys = [tree.predict(X) for tree in self.trees]

        Y = []
        conf = []
        for pred in range(len(Ys[0])):
            values = [Ys[tree][pred] for tree in range(len(Ys))]
            values = sorted(values)

            mode = values[0]
            max_count = 1
            current_count = 1
            for i in range(1, len(values)):
                if values[i] != values[i - 1]:
                    if current_count > max_count:
                        mode = values[i - 1]
                        max_count = current_count
                    current_count = 1
                else:
                    current_count += 1

            # Not exceeding check
            if current_count > max_count:
                mode = values[-1]
                max_count = max_count

            Y.append(mode)
            conf.append(max_count / self.num_trees)


        return (Y, conf)
