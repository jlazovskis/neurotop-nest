import numpy as np

# Load function for recursive simplex count.
# Shouldn't be dependent on where you run this script from.
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
module_path = (Path(__file__).parent / "recursive_maximal_count.py").absolute()
module_name = "rmc"
rmc_spec = spec_from_file_location(module_name, str(module_path))
rmc = module_from_spec(rmc_spec)
rmc_spec.loader.exec_module(rmc)
# ###################################

def _value_range(array):
    return np.max(array) - np.min(array)

def directionality(matrix):
    return np.sum(np.square(np.sum(matrix, axis = 0) - np.sum(matrix, axis = 1)))

def indegree_range(matrix):
    return _value_range(np.sum(matrix, axis = 0))

def outdegree_range(matrix):
    return _value_range(np.sum(matrix, axis = 1))

def degree_range(matrix):
    degrees = np.sum(matrix, axis = 0) + np.sum(matrix, axis = 1)
    return _value_range(degrees)

def maximal_simplex_count(matrix):
    return rmc.recursive_n_simplices(matrix)

def bidegree_range(matrix):
    bidegrees = np.sum(np.multiply(matrix, matrix.T), axis = 1)
    return _value_range(bidegrees)

def bidirectional_edges(matrix):
    return int(np.sum(np.multiply(matrix, matrix.T))/2)

def automorphisms():
    raise NotImplementedError
