from nsimplex import simulate
from itertools import product
import numpy as np
import pandas as pd
from scipy.special import comb
import argparse
from pathlib import Path
import pickle
from datetime import datetime

from utils.uniformity_measures import (
                          average_pearson,
                          pearson_range,
                          pearson_matrix,
                          average_cosine_distance,
                          average_pearson_directional,
                          spike_range,
                          spike_count,
                     )

from utils.structural import (
                          directionality,
                          indegree_range, outdegree_range,
                          bidegree_range, degree_range,
                          maximal_simplex_count,
                          bidirectional_edges
                     )
# Formatted printers
def ntnstatus(message):
    print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)


def ntnsubstatus(message):
    print('    '+message, flush=True)


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


# Database generation
def get_record(voltage, spike_trains, graph, id, seed, path):
    record = [
        id,
        seed,
        path,
        average_pearson(voltage),
        average_pearson(spike_trains),
        pearson_range(voltage),
        pearson_range(spike_trains),
        average_cosine_distance(voltage),
        average_cosine_distance(spike_trains),
        average_pearson_directional(voltage, graph),
        average_pearson_directional(spike_trains, graph),
        spike_range(spike_trains),
        spike_count(spike_trains),
        directionality(graph),
        indegree_range(graph),
        outdegree_range(graph),
        bidegree_range(graph),
        degree_range(graph),
        maximal_simplex_count(graph),
        np.log(maximal_simplex_count(graph)),
        bidirectional_edges(graph),
        pearson_matrix(voltage),
        pearson_matrix(spike_trains)
    ]
    return record


column_names = [
    'id',
    'seed',
    'path',
    'voltage PC',
    'ST PC',
    'voltage PC range',
    'ST PC range',
    'voltage cosine distance',
    'ST cosine distance',
    'directional voltage PC',
    'directional ST PC',
    'spike count range',
    'spike count',
    'directionality',
    'indegree range',
    'outdegree range',
    'bidegree range',
    'degree range',
    'maximal simplices',
    'log maximal simplices',
    'bidirectional edges',
    'voltage PC matrix',
    'ST PC matrix',
]


def get_sim_id(fname, n):
    combinations = int(n*(n-1)/2)
    simulation_id = fname[-combinations-20:-20]
    return simulation_id

def load_volts(voltage_path):
    volts_array = np.load(voltage_path, allow_pickle=True)
    volts = {'senders':volts_array[0], 'times':volts_array[1], 'V_m':volts_array[2]}
    return volts

def _volt_array(volt_dictionary, nnum):
    array = []
    for j in range(nnum):
        v_ind = np.where(volt_dictionary['senders'] == j+1)[0]
        array.append(volt_dictionary['V_m'][v_ind])
    return np.stack(array)

def load_spikes(spike_path):
    spikes_array = np.load(spike_path,
                      allow_pickle=True
                   )
    spikes = {'senders':spikes_array[0], 'times':spikes_array[1]}
    return spikes

def _spike_trains(spikes_dictionary, nnum, binsize, simlength):
    strains = []
    for j in range(nnum):
        try:
            sp_ind = np.where(spikes_dictionary['senders'] == j+1)[0]
            print(sp_ind)
            times = np.array(spikes_dictionary['times'][sp_ind])
            st = [np.count_nonzero(np.logical_and(times < node + binsize, times > node))
                  for node in list(range(0, simlength, binsize))]
        except TypeError: # Neuron had no spikes
            st = [0 for node in list(range(0, simlength, binsize))]
        strains.append(st)
    return np.stack(strains)

def build_df(args):
    simulations_root = Path(args.root + '/simulations/' + args.save_path).parent
    simulations_stem_prefix = Path(args.save_path).stem
    df = []
    df_path = Path(simulations_root / (simulations_stem_prefix + 'dataframe.pkl'))
    for voltage_path in simulations_root.glob(simulations_stem_prefix + '*volts.npy'):
        nnum = args.n
        simulation_id = get_sim_id(str(voltage_path), nnum)
        ntnstatus('Loading results of simulation '+simulation_id)
        spike_path = voltage_path.with_name(voltage_path.name.replace('volts','spikes'))
        structure_path = 'structure/' + args.structure_path + simulation_id + '.npy'
        volts = load_volts(voltage_path)
        volt_array = _volt_array(volts, nnum)
        spikes = load_spikes(spike_path)
        spike_trains = _spike_trains(spikes, nnum, args.binsize, args.time)
        graph = np.load(structure_path, allow_pickle = True)
        df.append(get_record(volt_array, spike_trains, graph, simulation_id, args.seed, str(voltage_path)))
    df = pd.DataFrame(df, columns = column_names)
    # Normalize directionality
    df1 = df[['bidirectional edges', 'directionality']].groupby(by='bidirectional edges').apply(lambda x: np.max(x) - np.min(x))
    df1 = df1.rename(columns = {'directionality':'directionality range'})
    df2 = df[['bidirectional edges', 'directionality']].groupby(by='bidirectional edges').apply(np.min)
    df2 = df2.rename(columns = {'directionality':'directionality min'})
    df = df.merge(df1, left_on = 'bidirectional edges', right_index = True, how = 'left')
    df = df.drop('bidirectional edges_y', axis = 1)
    df = df.rename(columns = {'bidirectional edges_x':'bidirectional edges'})
    df = df.merge(df2, left_on = 'bidirectional edges', right_index = True, how = 'left')
    df = df.drop('bidirectional edges_y', axis = 1)
    df = df.rename(columns = {'bidirectional edges_x':'bidirectional edges'})
    df['normalized directionality'] = (df['directionality'] - df['directionality min']) / df['directionality range']
    df = df.drop('directionality range', axis = 1)
    df = df.drop('directionality min', axis = 1)
    df[['normalized directionality']] = df[['normalized directionality']].fillna(value = 1)
    # Save
    df.to_pickle(df_path)
    return df


def run_simulations(args):
    ids = []
    save_parent = (Path("simulations") / args.save_path).parent
    save_parent.mkdir(exist_ok = True, parents = True)
    pickle.dump(vars(args), (Path("simulations") / (args.save_path + "args.pkl")).open('wb'))
    paths = build_matrices(Path(args.structure_path), args.n)
    for path in paths:
        args.exc_adj = str(path.with_name(path.stem))
        args.inh_adj = ''
        combinations = int(args.n * (args.n-1) / 2)
        args.id_prefix = path.stem[-combinations:]
        ids.append(simulate(args))
    df = build_df(args)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='n-simplex Nest Circuit Runner',
    usage='python nsimplex_runner.py')
    parser.add_argument('--n', type=int, default=3, help='Dimension of the simplex')
    parser.add_argument('--root', type=str, default='.', help='Root directory for importing and exporting files')
    parser.add_argument('--structure_path', type=str, default='3simplex/3simplex', help='Path to save circuit excitatory syn matrix, without .npy.')
    parser.add_argument('--save_path', type=str, default='3simplex/test', help='Path to save the results')
    parser.add_argument('--stimulus_targets', type=str, default="all", help='Stimulus targets. \'sink\', \'source\', \'all\' are supported')
    parser.add_argument('--stimulus_type', type=str, default="poisson", help='Stimulus type. \'dc\', \'ac\', \'poisson\', \'poisson_parrot\' are supported.')
    parser.add_argument('--stimulus_frequency', type=float, default=1., help='Stimulus frequency in ac case. Unusued for other stimuli.')
    parser.add_argument('--noise_strength', type=float, default=3., help='Strength of noise.')
    parser.add_argument('--stimulus_strength', type=int, default=40, help='Strength of stimulus.')
    parser.add_argument('--stimulus_length', type=int, default=100, help='Length of stimulus.')
    parser.add_argument('--stimulus_start', type=int, default=5, help='Length of stimulus.')
    parser.add_argument('--time', type=int, default=200, help='Length, in milliseconds, between stimuli. Must be an integer. Default is 200.')
    parser.add_argument('--threads', type=int, default=40, help='Number of parallel thread to use. Must be an integer. Default is 40.')
    parser.add_argument('--p_transmit', type=float, default=1, help='Synapse transmission probability. Default 1.')
    parser.add_argument('--seed', type=int, default=0, help='Simulaton seed. Default 0.')
    parser.add_argument('--binsize', type=int, default=5, help='Spike train bin size.')
    args = parser.parse_args()
    run_simulations(args)
