
import numpy as np
# Funcs to train and optimize network

def initialize_parameters(dim):
    """
    This function initializes the weights and bias.
    
    Arguments:
    dim -- size of the weights vector (number of features)
    
    Returns:
    w -- initialized weight vector of shape (dim, 1)
    b -- initialized bias (scalar)
    """
    w = np.zeros((dim, 1))  # Initializing the weights as zeros
    b = 0                   # Initializing the bias as zero
    return w, b

def sigmoid(z):
    """
    Compute the sigmoid of z.
    
    Arguments:
    z -- A scalar or numpy array of any size
    
    Returns:
    s -- sigmoid(z)
    """
    return 1 / (1 + np.exp(-z))

def propagate(w, b, X, y):
    """
    Implements the cost function and its gradient for the propagation step.
    
    Arguments:
    w -- weights, a numpy array of size (num_features, 1)
    b -- bias, a scalar
    X -- data of shape (num_features, num_samples)
    y -- true labels of shape (num_samples, 1)
    
    Returns:
    cost -- cost function value
    dw -- gradient of the loss with respect to weights w
    db -- gradient of the loss with respect to bias b
    """
    m = X.shape[1]  # Number of samples

    # FORWARD PROPAGATION
    z = np.dot(w.T, X) + b  # Linear step
    A = sigmoid(z)  # Activation function (sigmoid)
    cost = -1/m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))  # Cost function

    # BACKWARD PROPAGATION (derivatives)
    dw = 1/m * np.dot(X, (A - y).T)  # Gradient of the loss w.r.t w
    db = 1/m * np.sum(A - y)          # Gradient of the loss w.r.t b

    cost = np.squeeze(cost)  # Make sure cost is a scalar
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost

def optimize(w, b, X, y, learning_rate, print_cost=False):
    """
    Optimizes w and b by running gradient descent for one step.
    
    Arguments:
    w -- weights, a numpy array of size (num_features, 1)
    b -- bias, a scalar
    X -- data of shape (num_features, num_samples)
    y -- true labels of shape (num_samples, 1)
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the cost every 100 iterations
    
    Returns:
    w -- updated weights
    b -- updated bias
    cost -- the final cost after optimization
    """
    # Propagation step (forward + backward)
    grads, cost = propagate(w, b, X, y)
    
    # Retrieve derivatives from grads
    dw = grads["dw"]
    db = grads["db"]
    
    # Update rule: Gradient Descent
    w = w - learning_rate * dw  # Update weights
    b = b - learning_rate * db  # Update bias

    return w, b, cost