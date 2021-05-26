from nsimplex import simulate
from itertools import product
import numpy as np
from scipy.special import comb
import argparse
from pathlib import Path

def triu_from_array(array, n):
    matrix = np.zeros((n,n)).astype(array.dtype)
    counter = 0
    for i in range(n):
        for j in range(i+1,n):
            matrix[i,j] = array[counter]
            counter += 1
    return matrix

def build_matrices(path, n):
    save_paths = []
    for array in product([False,True], repeat = int(n*(n-1)/2)):
        matrix = triu_from_array(np.array(array), n)
        matrix = np.triu(np.ones((n,n)).astype(bool), 1)+matrix.T
        identifier = ''.join([str(int(elem)) for elem in array])
        save_path = path.with_name(path.stem + identifier + '.npy')
        (Path('structure') / save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(Path('structure') / save_path, matrix, allow_pickle = True)
        save_paths.append(save_path)
    return save_paths

def run_simulations(args):
    ids = []
    paths = build_matrices(Path(args.structure_path), args.n)
    for path in paths:
        args.exc_adj = str(path.with_name(path.stem))
        args.inh_adj = ''
        combinations = int(args.n * (args.n-1) / 2)
        args.id_prefix = path.stem[-combinations:]
        ids.append(simulate(args))
    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='n-simplex Nest Circuit Runner',
    usage='python nsimplex_runner.py')
    parser.add_argument('--n', type=int, default=3, help='Dimension of the simplex')
    parser.add_argument('--root', type=str, default='.', help='Root directory for importing and exporting files')
    parser.add_argument('--structure_path', type=str, default='3simplex/3simplex', help='Path to save circuit excitatory syn matrix, without .npy.')
    parser.add_argument('--save_name', type=str, default='3simplex', help='Path to save the results')
    parser.add_argument('--stimulus_targets', type=str, default="all", help='Stimulus targets. \'sink\', \'source\', \'all\' are supported')
    parser.add_argument('--stimulus_type', type=str, default="poisson", help='Stimulus type. \'dc\', \'ac\', \'poisson\', \'poisson_parrot\' are supported.')
    parser.add_argument('--stimulus_frequency', type=float, default=1., help='Stimulus frequency in ac case. Unusued for other stimuli.')
    parser.add_argument('--noise_strength', type=float, default=3., help='Strength of noise.')
    parser.add_argument('--stimulus_strength', type=int, default=40, help='Strength of stimulus.')
    parser.add_argument('--stimulus_length', type=int, default=100, help='Length of stimulus.')
    parser.add_argument('--stimulus_start', type=int, default=5, help='Length of stimulus.')
    parser.add_argument('--time', type=int, default=200, help='Length, in milliseconds, between stimuli. Must be an integer. Default is 200.')
    parser.add_argument('--threads', type=int, default=40, help='Number of parallel thread to use. Must be an integer. Default is 40.')
    args = parser.parse_args()
    print(run_simulations(args))
