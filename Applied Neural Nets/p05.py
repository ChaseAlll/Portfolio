import numpy as np
def mlp_check_dimensions(x, y, ws, bs):
    """
    Return True if the dimensions in double_u and beta agree.
    :param x: a list of lists representing the x matrix.
    :param y: a list output values.
    :param ws: a list of weight matrices (one for each layer)
    :param bs: a list of biases (one for each layer)
    :return: True if the dimensions of x, y, ws and bs match
    """
    ## W rows should equal X columns, b col should equal W col
    result = True
    if len(ws) != len(bs):
        return False
    if len(x[0]) != len(ws[0]):
        return False
    if len(x) != len(y):
        return False
    if len(y[0]) != len(bs[len(bs) - 1][0]):
        return False
    for layer in range(len(ws)):
        if len(ws[layer][0]) != len(bs[layer][0]):
            return False
        if layer == 0:
            pass
        else:
            prev_w = ws[layer - 1]
            if len(ws[layer]) != len(prev_w[0]):
                return False

        
    return result




def mlp_net_input(h, w, b):
    """
    Return the network input z as a function of h, w, and b.
    :param h: the input from the previous layer.
    :param w: the weights for this layer.
    :param b: the biases for this layer.
    :return: the linear network activations for this layer.
    """
    result = np.dot(h, w)
    result = np.add(result, b)
    return result
def mlp_tanh(z):
    """
    Return the hyperbolic tangent of z using numpy.
    :param z: the input "affinities" for this neuron.
    :return: hyperbolic tangent "squashing function
    """
    return np.tanh(z)
def mlp_softmax(z):
    """
    Return the softmax function at z using a numerically stable approach.
    :param z: A real number, list of real numbers, or list of lists of numbers.
    :return: The output of the softmax function for z with the same shape as z.
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
def mlp_feed_layer(h, w, b, phi):
    """
    Return the output of a layer of the network.
    :param h: The input to the layer (output of last layer).
    :param w: The weight matrix for this layer.
    :param b: The bias vector for this layer.
    :param phi: The activation function for this layer.
    :return: The output of this layer.
    """
    fuck = np.add(np.dot(h, w), b)
    result = phi(fuck)
    return result
def mlp_feed_forward(x, ws, bs,  phis):
    """
    Return the output of each layer of the network.
    :param x: The input matrix to the network.
    :param ws: The list of weight matrices for layers 1 to l.
    :param bs: The list of bias vectors for layers 1 to l.
    :param phis: The list of activation functions for layers 1 to l.
    :return: The list of outputs for layers 0 to l
    """
    h = x
    result = [[] for i in range(len(ws) + 1)]
    result[0] = h
    for layer in range(len(ws)):
        h = mlp_feed_layer(h, ws[layer], bs[layer], phis[layer])
        result[layer + 1] = h
    return result

def mlp_predict_proba(x, ws, bs,  phis):
    """
    Return the output matrix of probabilities for input matrix 'x'.
    :param x: The input matrix to the network.
    :param ws: The list of weight matrices for layers 1 to l.
    :param bs: The list of bias vectors for layers 1 to l.
    :param phis: The st matrix of probabilities (p)
    """
    pass
    ##each layer needs to be processed then passed through the net fucntion and then pushed through softmax
    prob = mlp_feed_forward(x, ws, bs, phis)
    return prob[len(prob) - 1]
def mlp_predict(x, ws, bs,  phis):
    """
    Return the output vector of labels for input matrix 'x'.
    :param x: The input matrix to the network.
    :param ws: The list of weight matrices for layers 1 to l.
    :param bs: The list of bias vectors for layers 1 to l.
    :param phis: The list of activation functions for layers 1 to l.
    :return: The output vector of class labels.
    """
    predict = mlp_predict_proba(x, ws, bs, phis)
    result = np.empty(len(predict))
    for row in range(len(predict)):
        result[row] = np.argmax(predict[row])
    return result

def mlp_data_prep():
    """
    Return the prepared data using the provided MNIST 1000 dataset.
    :return: The x matrix, y vector, and the wide matrix version of y
    """
    x_data = np.load('x_mnist1000.npy')
    y_data = np.load('y_mnist1000.npy') 
    y_matrix = np.zeros((len(y_data), 10))
    y_matrix[(range(len(x_data)), y_data)] = 1
    y_matrix = y_matrix.astype(int)
    np.random.seed(1)
    ran_index = np.random.permutation(range(0, len(x_data)))
    y_train_ran = [y_data[i] for i in ran_index]
    y_train_ran = np.asarray(y_train_ran)
    x_ran = [x_data[i] for i in ran_index]
    x_ran = np.asarray(x_ran)
    ##Y_test is not giving the right 5 values when printed [:5]
    num = 0
    #x_ran = np.random.permutation(x_data)
    y_matrix_ran = [y_matrix[i] for i in ran_index]
    y_matrix_ran = np.asarray(y_matrix_ran)
    """
    for index in range(len(ran_index)):
        y_train_ran[index] = (y_data[ran_index[index]])
        x_ran[index] = x_data[ran_index[index]]
        y_matrix_ran[index] = (y_matrix[ran_index[index]])
        num += 1
    """
    x_train = x_ran[0:800]
    y_matrix_train = y_matrix_ran[0:800]
    y_train = y_train_ran[0:800]
    x_test = x_ran[800:]
    y_matrix_test = y_matrix_ran[800:]
    y_test = y_train_ran[800:]
    x_train = np.transpose(x_train)
    x_test = np.transpose(x_test)
    for row in range(len(x_train)):
        mean = np.mean(x_train[row])
        std = np.std(x_train[row])
        if std == 0:
            std = 1
        for col in range(len(x_train[0])):
            add = np.subtract(x_train[row][col], mean)
            x_train[row][col] = np.divide(add,std)
            if col < 200:
                add2 = np.subtract(x_test[row][col], mean)
                x_test[row][col] = np.divide(add2, std)
    x_train = np.transpose(x_train)
    x_test = np.transpose(x_test)
    return x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test

############################################### Program 5 Start ##############################################################
def mlp_cost(x, y, ws, bs, phis, alpha):
    """
    Return the cross entropy cost function with L2 regularization term.
    :param x: a list of lists representing the x matrix.
    :param y: a list of lists of output values.
    :param ws: a list of weight matrices (one for each layer)
    :param bs: a list of biases (one for each layer)
    :param phis: a list of activation functions
    :param alpha: the hyperparameter controlling regularization
    :return: The cost function
    """
    # The L2 norm is the sum squared of the weights.
    sum_of_squares = 0
    for layer in range(len(ws)):
        buffer = np.power(ws[layer], 2)
        buffer = np.sum(buffer)
        sum_of_squares += buffer
    a = np.divide(alpha, 2)
    sum_of_squares = np.multiply(sum_of_squares, a)
    prob = mlp_predict_proba(x, ws, bs, phis)
    epsilon = .00000001
    prob = np.add(prob, epsilon)
    log = np.log(prob)
    product = np.multiply(y, log)
    sum = np.sum(product)
    denom = np.divide(-1, len(x))
    result = np.multiply(sum, denom)
    result = np.add(result, sum_of_squares)
    return result
def mlp_propagate_error(x, y, ws, bs, phis, hs):
    """
    Return a list containing the gradient of the cost with respect to z^(k)for each layer.
    :param x: a list of lists representing the x matrix.
    :param y: a list of lists of output values.
    :param ws: a list of weight matrices (one for each layer)
    :param bs: a list of biases (one for each layer)
    :param phis: a list of activation functions
    :param hs: a list of outputs for each layer include h^(0) = x
    :return: A list of gradients of J with respect to z^(k) for k=1..l
    """
    #Shape mismatch on single iteration, like the 4th. It is the multiply function. 
    #thing needs to use whatever that layer it is. 
    P = hs[len(hs) - 1]
    diff = np.subtract(P, y)
    D_l = np.divide(diff, len(x))
    result = []
    result.append(D_l)
    D_k = np.copy(D_l)
    for layers in range((len(ws) - 1) ,0, -1):
        thing = np.power(hs[layers], 2)
        thing = np.subtract(1, thing)
        transpose = np.transpose(ws[layers])
        D_k = np.dot(D_k, transpose)
        D_k = np.multiply(D_k, thing)
        result.append(D_k)
    result.reverse()
    return result
def mlp_gradient(x, y, ws, bs, phis, alpha):
    """
    Return a list containing the gradient of the cost with respect to z^(k)for each layer.
    :param x: a list of lists representing the x matrix.
    :param y: a list of lists of output values.
    :param ws: a list of weight matrices (one for each layer)
    :param bs: a list of biases (one for each layer)
    :param phis: a list of activation functions:param hs
    : a list of outputs for each layer include h^(0) = x
    :return: A list of gradients of J with respect to z^(k) for k=1..l
    """
    hs = mlp_feed_forward(x, ws, bs, phis)
    D = mlp_propagate_error(x, y, ws, bs, phis, hs)
    result_w = []
    result_b = []
    w_1 = np.dot(np.transpose(x), D[0])
    step = np.multiply(alpha, ws[0])
    w_1 = np.add(w_1, step)
    w_1 = np.ndarray.tolist(w_1)
    result_w.append(w_1)
    for layers in range(1, len(ws)):
       w_2 = np.dot(np.transpose(hs[layers]), D[layers])
       w_2 = np.add(w_2, np.multiply(alpha, ws[layers]))
       result_w.append(w_2)
    for layers in range(len(ws)):
        ones = np.ones((len(x), 1))
        b_1 = np.dot(np.transpose(ones), D[layers])
        result_b.append(b_1)
    result_w = np.reshape(result_w, (1, -1))
    return result_w, result_b
def mlp_initialize(layer_widths):
    """
    Use Numpy's random package to initialize a list of weights,a list of biases, 
    and a list of activation functions forthe number of nodes per layer provided in the argument.
    To pass the tests you will need to initialize the matricesin the following order:ws1, bs1, ws2, bs2, ..., wsl, bsl.
    
    :param layer_widths: a list of layer widths
    :return: a list of weights, a list of biases, and a list ofphis, one for each layer
    """
    #weight dimension should be first argument by second argument and increment. bias should be 1 by whatever starting on the second size.
    #Need to fix for edge case, it is not initializing any values for a single layer network or two layer. Now main problem.
    #Need to also complete the edge case for last layer
    result_w = []
    result_b = []
    result_phi = []
    length = len(layer_widths)
    for layer in range(length):
        if layer < len(layer_widths) - 1:
            result_w.append(np.random.normal(0, 0.1, (layer_widths[layer], layer_widths[layer + 1]) ))
            result_b.append(np.random.normal(0, 0.1, (1, layer_widths[layer + 1])))
    for layer in range(length - 1):
        if layer == length - 2:
            result_phi.append(mlp_softmax)
        else:
            result_phi.append(mlp_tanh)
    if length == 1:
        result_w.append(np.random.normal(0, 0.1, (784 , layer_widths[0])))
        result_b.append(np.random.normal(0, 0.1, (1, layer_widths[0])))
        result_phi.append(mlp_softmax)
    """
    else:
        result_w.append(np.random.normal(0, 0.1, (layer_widths[len(layer_widths) - 1], 10)))
        result_b.append(np.random.normal(0, 0.1, (1, 10)))
    """
    return result_w, result_b, result_phi
    
def mlp_gradient_descent(x, y, ws0, bs0, phis, alpha, eta, n_iter):
    """Uses gradient descent to estimate the weights, ws, and biases, bs,that reduce the cost.
    :param x: a list of lists representing the x matrix.
    :param y: a list of lists of output values.
    :param ws0: a list of initial weight matrices (one for each layer)
    :param bs0: a list of initial biases (one for each layer)
    :param phis: a list of activation functions
    :param alpha: the hyperparameter controlling regularization
    :param eta: the learning rate
    :param n_iter: the number of iterations
    :return: the estimate weights, the estimated biases
    """
    #cost function works, updating weights / biases has something wrong. 
    weights = ws0.copy()
    bias = bs0.copy()
    for index in range(n_iter):
        cost = mlp_cost(x, y, weights, bias, phis, alpha)
        print(cost)
        w_gradient, b_gradient = mlp_gradient(x, y, weights, bias, phis, alpha)
        for layers in range(len(weights)):
            weight_product = np.multiply(eta, w_gradient[layers])
            weights[layers] = np.subtract(weights[layers], weight_product)
            bias_product = np.multiply(eta, b_gradient[layers])
            bias[layers] = np.subtract(bias[layers], bias_product)
    return weights, bias
def mlp_run_mnist():
    """
    Prepare the data from the local directory and run gradient descentto estimate the parameters on the training data.  
    Use a learning rate of 0.2, regularization term of 0.05, 450 nodes in a singlehidden layer, and 300 iterations of gradient descent.
    
    :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
    """
    # issue with dimensions when computing the elementwise product between y and the log of the probabilities of x
    x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_data_prep()
    eta = 0.2
    alpha = 0.05
    n_iter = 300
    layer_widths = [784, 450, 10]
    ws0, bs0, phis = mlp_initialize(layer_widths)
    ws_hat, bs_hat = mlp_gradient_descent(x_train, y_matrix_train, ws0, bs0, phis, alpha, eta, n_iter)
    y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis)
    train_acc = (y_hat_train == y_train).mean()
    y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis)
    test_acc = (y_hat_test == y_test).mean()
    return x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_acc, test_acc

np.random.seed(1)
m, n0, n1, n2 = 10, 8, 5, 3
x = 2 * np.random.rand(m, n0) - 1
w1 = np.random.randn(n0, n1)
b1 = 0.1*np.random.randn(1, n1)
w2 = np.random.randn(n1, n2)
b2 = 0.1*np.random.randn(1, n2)
phi1 = mlp_tanh
phi2 = mlp_softmax
ws = [w1, w2]
bs = [b1, b2]
phis = [phi1, phi2]
p = mlp_predict_proba(x, ws, bs, phis)
r = np.random.rand(m, 1)
y = np.argmax(p.cumsum(axis=1) > r, axis=1)
y_matrix = np.zeros((m, n2))
y_matrix[(range(m), y)] = 1
alpha = 0.01
hs = mlp_feed_forward(x, ws, bs, phis)
ds = mlp_propagate_error(x, y_matrix, ws, bs, phis, hs)

x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0,ws_hat, bs_hat, train_acc, test_acc = mlp_run_mnist()
print(ws_hat[0])
print(ws_hat[1])
print(bs_hat[0])
print(bs_hat[1])
print(train_acc)
print(test_acc)