import numpy as np

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

