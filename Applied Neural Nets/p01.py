
def dot_product(u, v):
    """
    return the vector dot-product between equal length vectors.

    :param u: list of numbers(vector)
    :param v: list of numbers(vector)
    :return: the dot-product
    """
    i = 0
    result = 0
    for index in u:
      result += index * v[i]
      i += 1
    return result
def matrix_multiply(a, b):
    """
    Return the matrix product of a and b.

    :param a: the left matrix operand
    :param b: the right matrix operand
    :return: the matrix product of a and b

    """
    result = [[0 for x in range(len(b[0]))] for x in range(len(a))] 
    for a_row in range(len(a)):
        for b_row in range(len(b[0])):
            for b_col in range(len(b)):
                result[a_row][b_row] = round(result[a_row][b_row] + a[a_row][b_col] * b[b_col][b_row], 6)
    return result



def transpose(a): 
    """
    Returns the matrix transpose of 'a'.

    :param a: The original matrix.
    :return: the transposed matrix. 

    """
    result = [[0] * len(a) for x in range(len(a[0]))] 
    for a_row in range(len(a)):
        for a_col in range(len(a[a_row])):
            result[a_col][a_row] = a[a_row][a_col]
    return result

def trace(a):
    """
    Returns the race of matrix 'a'.

    :param a: The matrix. 
    :return: The trace of the matrix

    """
    result = 0
    for index in range(len(a)):
        try:
            result += a[index][index]
        except:
            return result
    return result

def inverse2d(a):
    """
    Returns the matrix inverse of 2x2 matrix "a".

    :param a: The original 2x2 matrix.
    :return: Its matrix inverse. 

    """
    result = [[0, 0], [0, 0]]
    det_a = (a[1][1] * a[0][0]) - (a[0][1] * a[1][0])
    for a_row in range(len(result)):
        for a_col in range(len(result[a_row])):
           result[a_row][a_col] = round(a[a_row][a_col] * (1/det_a), 6)
    buffer = result[1][1]
    result[0][1] *= -1
    result[1][0] *= -1
    result[1][1] = result[0][0]
    result[0][0] = buffer
    return result
    
def pnorm(x, p):
    """
    Returns the L_p norm of vector 'x'. 

    :param x: The vector.
    :param p: The order of the norm.
    :return: The L_p norm of the matrix. 

    """
    result = 0
    for index in x:
        result += abs(index) ** p
    result = result ** (1/p)
    return result
print(inverse2d([[0.122807, -0.017544], [0.017544, 0.140351]]))