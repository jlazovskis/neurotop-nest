import numpy as np


def get_sig(matrix):
    sig = 0
    c = 0
    for i, row in enumerate(matrix):
        for j in row[i+1:]:
            sig += 2**c if j else 0
            c += 1
    return int(sig + 2**c)


def final_indices(matrix):
    indices = []
    for i in range(matrix.shape[0]):
        j = i+1
        while j < matrix.shape[0]:
            if not matrix[i, j]:
                break
            j = j+1
        if j >= matrix.shape[1]:
            indices.append(i)
    return indices


def submatrix(matrix, y):
    return np.delete(np.delete(matrix, y, axis=0), y, axis=1)


matrix_table = {}


def recursive_n_simplices(matrix, hashing=True):
    # Small wrapper to convert to used format: triu matrix with entries equal to bedges
    _matrix = np.tril(matrix).T
    return _recursive_n_simplices(_matrix)


def _recursive_n_simplices(_matrix, hashing=True):
    if hashing:
        try:
            return matrix_table[get_sig(_matrix)]
        except KeyError:
            if _matrix.shape[0] == 1:
                matrix_table[get_sig(_matrix)] = 1
                return 1
            res = np.sum([_recursive_n_simplices(submatrix(_matrix, y), hashing=True) for y in final_indices(_matrix)])
            matrix_table[get_sig(_matrix)] = res
            return res
    else:
        if _matrix.shape[0] == 1:
            return 1
        return np.sum([_recursive_n_simplices(submatrix(_matrix, y), hashing=False) for y in final_indices(_matrix)])
