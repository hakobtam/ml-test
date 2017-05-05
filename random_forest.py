from decision_tree import DecisionTree
from collections import defaultdict
import numpy as np

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
        self.trees = None
        self.ratio_per_tree = ratio_per_tree

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        # TODO:Build self.num_trees trees of depth self.max_tree_depth
        # with randomized data.
        # TODO: Remove this toto and the todo above after you
        # implement the todo above.
        self.trees = []
        batch_size = int(len(Y) * self.ratio_per_tree)
        X = np.array(X)
        Y = np.array(Y)
        data = np.column_stack((X,Y))
        for i in range(self.num_trees):
            np.random.shuffle(data)
            tree = DecisionTree(self.max_tree_depth)
            tree.fit(data[:batch_size,:-1], data[:batch_size,-1])
            self.trees.append(tree)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        Y = []
        conf = []
        label_count = defaultdict(int)
        for ix in range(len(X)):
            label_dict = defaultdict(int)
            for it in range(self.num_trees):
                label_dict[self.trees[it].predict(X)[ix]] += 1
            label_count = np.array(list(label_dict.values()))
            count = np.max(label_count)
            index = np.argmax(label_count)
            label = list(label_dict.keys())[index]
            
            Y.append(label)
            conf.append(count)

        # TODO: Evaluate labels in each of the `self.tree`s and return the
        # label and confidence with the most votes for each of
        # the data points in `X`
        # TODO: Remove this toto and the todo above after you
        # implement the todo above.
        return (Y, conf)
