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

x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_data_prep()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_matrix_train.shape)
print(y_matrix_test.shape)
print(x_train[:10, 200:205])
print(x_test[:10, 200:205])
print(y_train[:5])
print(y_test[:5])
print(y_matrix_train[:5, :])
print(y_matrix_test[:5, :])
"""
m, n_0, n_1, n_2 = np.random.randint(2, 10, 4)
x = np.zeros((m, n_0))
y = np.zeros((m, n_2))
w_1 = np.zeros((n_0, n_1))
w_2 = np.zeros((n_1, n_2))
b_1 = np.zeros((1, n_1))
b_2 = np.zeros((1, n_2))
double_u = [w_1, w_2]
beta = [b_1, b_2]
print(beta)
"""