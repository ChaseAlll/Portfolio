import numpy as np
def affinity(x, w):
    """
    Returns the affinity of each row of 'x' for each column of 'w'.
    :param x: List of lists representing X matrix, each row is an x^(i) vector.
    :param w: List of lists representing the weights (one column for each class)..
    :return: A numpy array of affinities.
    """
    return np.dot(x, w)
def softmax_function(z):
    """
    Return the softmax_function at z using a numerically stable approach.
    :param z: A real number, list of numbers, or list of lists of numbers.
    :return: The output of the softmax_function function for z withthe same shape as the input.
    """
    index = np.argmax(z)
    result = np.empty((len(z), len(z[0])))
    copy = np.copy(z)  
    copy = np.subtract(copy, copy.item(index))
    for row in range(len(z)):
        denom = 0
        xp_1 = copy[row]
        for col in range(len(z[0])):
            xp_2 = copy[row][col]
            denom += np.exp(xp_2)
        buffer = (np.divide(np.exp(xp_1), denom))
        result[row] = buffer
    return result
def softmax_predict_proba(x, w):
    """
    Return the estimated probability of each class for each row of 'x'.
    :param x: List of lists representing X matrix, each row is an x^(i) vector.
    :param w: List of lists representing the weights (one column for each class)..
    :return: A numpy 2d array of probabilities.
    """
    return softmax_function(affinity(x,w))
def softmax_predict(x, w):
    """
    Return the estimated label for each row of 'x'.
    :param x: List of lists representing X matrix, each row is an x^(i) vector.
    :param w: List of lists representing the weights (one column for each class)..
    :return: A numpy array of class labels.
    """
    predict = softmax_predict_proba(x, w)
    result = np.empty(len(predict))
    for row in range(len(predict)):
        result[row] = np.argmax(predict[row])
    return result

def softmax_cost(x, y, w):
    """
    Return the cross entropy cost for softmax regression.
    :param x: List of lists representing X matrix, each row is an x^(i) vector.
    :param y: List representing the y vector of labels.
    :param w: List of lists representing W matrix, each column contains the weightvector for a class.
    :return: The value of the cost function.
    """
    prob = softmax_predict_proba(x, w)
    Y_indexes = softmax_predict(x, w)
    Y = np.zeros((len(prob), len(prob[0])))
    for col in range(len(Y_indexes)):
        index = y[col]
        Y[col][index] = 1
    log = np.log(prob)
    product = np.multiply(Y, log)
    sum = np.sum(product)
    denom = np.divide(-1, len(x))
    result = np.multiply(sum, denom)
    return result
def softmax_gradient(x, y, w):
    """
    Return the gradient of the cross entropy cost function.
    :param x: List of lists representing X matrix, each row is an x^(i) vector.
    :param y: List of correct y-values [0... c-1]:param w: List of lists of weights.  Each column contains the weights for oneclass.
    :return: The gradient of the cross entropy cost function as a list of lists or2D numpy array.
    """
    denom = np.divide(1, len(x))
    P = softmax_predict_proba(x, w)
    X_t = np.transpose(x)
    Y = np.zeros((len(P), len(P[0])))
    for col in range(len(x)):
        index = y[col]
        Y[col][index] = 1
    subtraction = np.subtract(P, Y)
    product = np.dot(X_t, subtraction)
    result = np.multiply(denom, product)
    return result 
def softmax_gradient_descent(x, y, w_init, eta, n_iter):
    """
    Uses gradient descent to estimate the weights 'w' that reduce thelogistic regression cost function.
    :param x: List of lists representing X matrix, each row is an x^(i) vector.
    :param y: List of correct labels [0... c-1].:param w_init: List of lists of initial weights.
    :param eta: The learning rate.
    :param n_iter: The number of parameter updates to perform.
    :return: A 2D numpy array of the estimated weights.
    """
    W = np.copy(w_init)
    for index in range(n_iter):
        print(softmax_cost(x, y, W))
        gradient = softmax_gradient(x, y, W)
        product = np.multiply(eta, gradient)
        W = np.subtract(W, product)
    return W
np.random.seed(1)
n, m, c = 10, 4, 3
w = 10 * np.random.randn(m, c)
x = 2 * np.random.rand(n, m) - 1
y = [np.argmax(np.random.multinomial(1, softmax_predict_proba(x[[i]], w)[0]))
for i in range(n)]
w_init = np.random.normal(0, 0.1, (m, c))
eta = 1
n_iter = 1000
print(softmax_gradient_descent(x, y, w_init, eta, n_iter))