import numpy as np
import recursive_maximal_count as rmc

def _value_range(array):
    return np.max(array) - np.min(array)

def directionality(matrix):
    return np.sum(np.square(np.sum(matrix, axis = 0) - np.sum(matrix, axis = 1))

def indegree_range(matrix):
    return _value_range(np.sum(matrix, axis = 0))

def outdegree_range(matrix):
    return _value_range(np.sum(matrix, axis = 1))

def degree_range(matrix):
    degrees = np.sum(matrix, axis = 0) + np.sum(matrix, axis = 1)
    return _value_range(degrees)

def maximal_simplex_count(matrix):
    return rmc.recursive_n_simplex(matrix)

def bidegree_range(matrix):
    bidegrees = np.sum(np.multiply(matrix, matrix.T), axis = 1)
    return _value_range(bidegrees)

def bidiretional_edges(matrix):
    return int(np.sum(np.multiply(matrix, matrix.T))/2)

def automorphisms:
    raise NotImplementedError
