from dtOmitted import build_tree
import numpy as np

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """
    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        XX = np.array(X)
        YY = np.array(Y)
        data = np.column_stack((XX,YY))
        self.trees = build_tree(data, 0, self.max_depth)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        Y = []
        for x in X:
            tree_node = self.trees
            current_depth = 0
            while tree_node.is_leaf == False and current_depth != self.max_depth:
                current_depth += 1
                if x[tree_node.column] >= tree_node.value:
                    tree_node = tree_node.true_branch
                else:
                    tree_node = tree_node.false_branch
            Y.append(tree_node.result)

        return Y
