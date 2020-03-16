import numpy as np
import math
from scipy.linalg import toeplitz

"""
Based on https://github.com/alisaaalehi/convolution_as_multiplication with minor changes for our HW
"""

def matrix_to_vector(input):
    """
    Converts the input matrix to a vector by stacking the rows in a specific way explained here

    Arg:
    input -- a numpy matrix

    Returns:
    ouput_vector -- a column vector with size input.shape[0]*input.shape[1]
    """
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h * input_w, dtype=input.dtype)
    # flip the input matrix up-down because last row should go first
    input = np.flipud(input)
    for i, row in enumerate(input):
        st = i * input_w
        nd = st + input_w
        output_vector[st:nd] = row
    return output_vector


def vector_to_matrix(input, output_shape):
    """
    Reshapes the output of the maxtrix multiplication to the shape "output_shape"

    Arg:
    input -- a numpy vector

    Returns:
    output -- numpy matrix with shape "output_shape"
    """
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i * output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    # flip the output matrix up-down to get correct result
    output = np.flipud(output)
    return output


def generateRj(I, F, print_ir=False):
    # number of columns and rows of the input
    I_row_num, I_col_num = I.shape

    # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1
    if print_ir: print('output dimension:', output_row_num, output_col_num)

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                               (0, output_col_num - F_col_num)),
                           'constant', constant_values=0)
    if print_ir: print('F_zero_padded: ', F_zero_padded)

    # use each row of the zero-padded F to creat a toeplitz matrix.
    #  Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(F_zero_padded.shape[0] - 1, -1, -1):  # iterate from last row to the first row
        c = F_zero_padded[i, :]  # i th row of the F
        r = np.r_[c[0], np.zeros(I_col_num - 1)]  # first row for the toeplitz fuction should be defined otherwise
        # the result is wrong
        toeplitz_m = toeplitz(c, r)  # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)
        if print_ir: print('F ' + str(i) + '\n', toeplitz_m)

        # doubly blocked toeplitz indices:
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0] + 1)
    r = np.r_[c[0], np.zeros(I_row_num - 1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    if print_ir: print('doubly indices \n', doubly_indices)

    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape  # shape of one toeplitz matrix
    h = toeplitz_shape[0] * doubly_indices.shape[0]
    w = toeplitz_shape[1] * doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape  # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i, j] - 1]

    # # get result of the convolution by matrix mupltiplication
    # tmp = []
    # for i in range(73, 434, 23):  # range(73, 434, 23) (95, 455, 23) q_size=k_size=8,r_size=16 TODO
    #     for j in range(16):
    #         tmp.append(doubly_blocked[i + j])
    # tmp = np.array(tmp)
    # return tmp

    # tmp = [] #q_size=k_size=5,r_size=10
    # for i in range(30, 166, 14):
    #     for j in range(10):
    #         tmp.append(doubly_blocked[i + j])
    # tmp = np.array(tmp)
    # return tmp

    # tmp = []  # q_size=5, r_size=10, k_size=9
    # for i in range(76, 248, I_row_num + F_row_num - 1):
    #     for j in range(F_row_num):
    #         tmp.append(doubly_blocked[i + j])
    # tmp = np.array(tmp)
    # return tmp
    Rj = []
    full_conv_size = I_row_num + F_row_num - 1
    s = (full_conv_size - F_row_num)
    start = int(math.floor(s / 2)) * full_conv_size + int(math.ceil(s / 2))
    idx = start
    while len(Rj) < F_row_num ** 2:
        for j in range(F_row_num):
            Rj.append(doubly_blocked[idx + j])
        idx += full_conv_size
    Rj = np.array(Rj)
    return Rj
