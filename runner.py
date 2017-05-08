import numpy as np
import matplotlib.pyplot as plt

from decision_tree import DecisionTree
from random_forest import RandomForest
from logistic_regression import sigmoid
from logistic_regression import normalized_gradient
from logistic_regression import gradient_descent


def accuracy_score(Y_true, Y_predict):

    acc = 0.
    for y_true, y_predicted in zip(Y_true, Y_predict):
        acc += (y_true == y_predicted)

    return acc * 100. / len(Y_true)


def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    Y = np.array([data[:, 0]]).T
    n, d = X.shape

    tree_accuracies = []
    log_accuracies = []
    forest_accuracies = []

    for trial in range(15):
        # TODO: shuffle for each of the trials.
        # the following code is for reference only.
        idx = np.arange(n)
        np.random.seed(2017)
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]
             
        msk = np.random.rand(len(X)) <= 0.75
        Xtrain = X[msk]
        Xtest  = X[~msk]   
        Ytrain = Y[msk]
        Ytest  = Y[~msk]

        # train the decision tree
        classifier = DecisionTree(100)
        print ("LOG 1")
        classifier.fit(Xtrain, Ytrain)
        
        # output predictions on the remaining data
        y_pred = classifier.predict(Xtest)
        accuracy = accuracy_score(Ytest, y_pred)
        print("accuracy of decision tree: ", accuracy_score(Ytest, y_pred))

        #Random forest part
        classifier = RandomForest(30,100)
        #accuracy_score(Ytest, y_pred)
        #classifier.fit(Xtrain, Ytrain)

        #accuracies = [accuracy_score(Ytest,tree.predict(Xtest)) for tree in classifier.trees]
        #print(accuracies)
        print ("LOG 2")
        classifier.fit(Xtrain, Ytrain)

        y_pred = classifier.predict(Xtest)[0]
        forest_accuracies.append(accuracy_score(Ytest, y_pred))
        print("accuracy of forest: ", accuracy_score(Ytest, y_pred))

        dd = np.array(Xtrain)
        data_train = np.column_stack((np.ones(dd.shape[0]), dd))
        label_train = [-1 if a == 0 else 1 for a in Ytest]
        beta_hat = gradient_descent(data_train, label_train, epsilon=1e-3, l=1, step_size=0.1, max_steps=200)
        
        y_pred = [1 if a >= 0 else -1 for a in data_train.dot(beta_hat)]
        log_accuracies.append(accuracy_score(label_train, y_pred))
        print("accuracy of logistic: ", accuracy_score(label_train, y_pred))

    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(tree_accuracies)
    stddevDecisionTreeAccuracy = np.std(tree_accuracies)

    meanLogisticRegressionAccuracy = np.mean(log_accuracies)
    stddevLogisticRegressionAccuracy = np.std(log_accuracies)

    meanRandomForestAccuracy = np.mean(forest_accuracies)
    stddevRandomForestAccuracy = np.std(forest_accuracies)

    stats = np.zeros((3, 3))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluate_performance()
    print ("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print ("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print ("Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
