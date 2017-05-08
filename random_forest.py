import numpy as np
from decision_tree import DecisionTree
from collections import defaultdict

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
        self.trees = None

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        self.trees = []
        total_size = int(len(Y))
        batch_size = int(total_size* self.ratio_per_tree)
        
        X_features = np.array(X)
        Y_classes = np.array(Y)
        data = np.column_stack((X_features,Y_classes))
        
        for i in range(self.num_trees):
            #1) shuffling the data with only order changes
            np.random.shuffle(data)
            
            #2) building the tree
            tree = DecisionTree(self.max_tree_depth) #Getting max_tree_depth in classes init
            tree.fit(data[:batch_size,:-1], data[:batch_size,-1]) #building the actuall tree
            
            #3) Adding the newly built tree to the forest
            self.trees.append(tree)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        Y = []
        confidence = []
        
        toatal_label_count = defaultdict(int)
        
        for row in range(len(X)):
            label_dict = defaultdict(int)
            
            for tree in range(self.num_trees):
                #lets see how each tree predicts this
                label_dict[self.trees[tree].predict(X)[row]] += 1
                
            toatal_label_count = np.array(list(label_dict.values()))
            count = np.max(toatal_label_count)
           # confidence_single_tree = float(count/len(toatal_label_count))
            
            index = np.argmax(toatal_label_count)
            label = list(label_dict.keys())[index]
            
            Y.append(label)
            confidence.append(count)
        
        return (Y, confidence)
