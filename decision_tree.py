#!/usr/bin/python3

"""
this module contains the DecisionTree class
"""

from dtOmitted import build_tree

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """

    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth
        self.tree = None

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        if not isinstance(X, list):
            X = X.tolist()
        if not isinstance(Y, list):
            Y = Y.tolist()

        data = [X[i][:] + [Y[i]] for i in range(len(X))]
        self.tree = build_tree(data, max_depth=self.max_depth)


    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        if not isinstance(X, list):
            X = X.tolist()

        return [self.predict_one(self.tree, row) for row in X]

    def predict_one(self, node, row):
        """
        predicts the label recursively for just one piece of data
        :param node: current node in the decision tree, instance of DecisionNode
        :param x: the object to classify, list
        """
        if not node.is_leaf:
            x_value = row[node.column]
            value = node.value
            if isinstance(value, (int, float)):
                if x_value > value:
                    return self.predict_one(node.true_branch, row)
                return self.predict_one(node.false_branch, row)
            else:
                if x_value == value:
                    return self.predict_one(node.true_branch, row)
                return self.predict_one(node.false_branch, row)
        return node.results

