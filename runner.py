import numpy as np
import matplotlib.pyplot as plt
from random_forest import RandomForest
from decision_tree import DecisionTree


def accuracy_score(Y_true, Y_predict):
    acc = 0.
    for y_t, y_p in zip(Y_true, Y_predict):
        #print(y_t, ' ', y_p)
        acc += (y_t == y_p)
    return acc * 100. / len(Y_true)

def sigmoid(s):
    return 1.0/(1.0 + np.exp(-s))

def normalized_gradient(X, Y, beta, l):
    return np.sum(np.array([-Y[i] * X[i] * (1 - sigmoid(Y[i]*(beta.T.dot(X[i])))) for i in range(len(Y))]), axis = 0)/len(Y) + l * beta/len(Y)

def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):

    beta = np.zeros(X.shape[1])
    mean_val = np.mean(X[:,1:], axis = 0)
    sigma = np.std(X[:,1:], axis = 0) 
    lam = np.hstack((0, l / (sigma ** 2)))
    mean_val = np.hstack((0, mean_val))
    sigma = np.hstack((1, sigma))
    X_scale = (X - mean_val) / sigma
    step_size = 0.1
    beta = np.zeros((X.shape[1]))
    for s in range(max_steps):
        grad = normalized_gradient(X_scale, Y, beta, lam)
        beta = beta - step_size * grad
        if np.linalg.norm(step_size * grad) / np.linalg.norm(beta) < epsilon:
            break
    beta[0] = beta[0] - np.sum((mean_val * beta) / sigma)
    beta[1:] = beta[1:] / sigma[1:]
    return beta


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
    filename = 'SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape
    tree_accuracies = []
    log_accuracies = []
    forest_accuracies = []
    for trial in range(7):
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        Xtrain = X[:101, :]
        Xtest = X[101:, :]
        ytrain = y[:101, :]
        ytest = y[101:, :]

        classifier = DecisionTree(3)
        classifier.fit(Xtrain, ytrain)

        y_pred = classifier.predict(Xtest)
        tree_accuracies.append(accuracy_score(ytest, y_pred))
        #print("acc of decision tree= ", accuracy_score(ytest, y_pred))

        classifier = RandomForest(27,100)
        classifier.fit(Xtrain, ytrain)

        y_pred = classifier.predict(Xtest)[0]
        forest_accuracies.append(accuracy_score(ytest, y_pred))
        #print("acc of forest= ", accuracy_score(ytest, y_pred))

        dd = np.array(Xtrain)
        data_train = np.column_stack((np.ones(dd.shape[0]), dd))
        label_train = [-1 if a == 0 else 1 for a in ytrain]
        beta_hat = gradient_descent(data_train, label_train, epsilon=1e-3, l=1, step_size=0.1, max_steps=200)
        
        y_pred = [1 if a >= 0 else -1 for a in data_train.dot(beta_hat)]
        log_accuracies.append(accuracy_score(label_train, y_pred))
        #print("acc of logistic= ", accuracy_score(label_train, y_pred))

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
