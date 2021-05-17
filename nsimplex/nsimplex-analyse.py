# N-dimensional simplices analysis tool
# Date: April 2021

# Packages
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.sparse as sparse
import sys                                                                  # For current directory to read and write files
import argparse                                                             # For options
from datetime import datetime                                               # For giving messages
import random
import matplotlib.pyplot as plt                                             # For plotting
import matplotlib as mpl                                                    # For plotting

from pathlib import Path                                                    # For file management


from utils.uniformity_measures import (
                          average_pearson,
                          average_cosine_distance,
                          average_pearson_directional,
                          spike_range,
                          spike_count
                     )
from utils.structural import (
                          directionality,
                          indegree_range, outdegree_range,
                          bidegree_range, degree_range,
                          maximal_simplex_count,
                          bidirectional_edges
                     )

#******************************************************************************#
# Formatted printer for messages
def ntnstatus(message):
    print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)
def ntnsubstatus(message):
    print('    '+message, flush=True)


#******************************************************************************#
# Data loading utils
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
        sp_ind = np.where(spikes_dictionary['senders'] == j+1)[0]
        times = np.array(spikes['times'][sp_ind])
        st = [np.count_nonzero(np.logical_and(times < node + binsize, times > node))
                  for node in list(range(0, simlength, binsize))[:-1]]
        strains.append(st)
    return np.stack(strains)

#******************************************************************************#
# Traces plotter
def plot_traces(volts, spikes, figname):
    ntnstatus("Plotting results for " + str(figname) + " ...")
    # Style
    c_exc = (0.8353, 0.0, 0.1961)
    c_inh = (0.0, 0.1176, 0.3843)
    color_list = [c_exc for sender in spikes['senders']]

    # Set up figure
    fig, ax = plt.subplots(figsize=(20,6))
    mpl.rcParams['axes.spines.right'] = False; mpl.rcParams['axes.spines.top'] = False
    plt.xlim(0,args.time)
    ax.set_ylim(0,nnum+.5)
    ax.set_yticks([i+1 for i in range(nnum)])
    ax.invert_yaxis()

    # Plot individual voltage
    v_min = min(volts['V_m'])
    v_max = max(volts['V_m'])
    v_range = max([0.1,v_max-v_min])
    for j in range(1, nnum+1):
        v_ind = np.where(volts['senders'] == j)[0]
        sp_ind = np.where(spikes['senders'] == j)[0]
        ax.plot([volts['times'][i] for i in v_ind], [-(volts['V_m'][i]-v_min)/v_range+j for i in v_ind])
        ax.annotate(str(len(
                    [spikes['times'][i] for i in sp_ind]
                )), [5,j])

    # Plot individual spikes
    ax.scatter(spikes['times'], spikes['senders'], s=20, marker="s",  edgecolors='none', alpha=.8)
    ax.set_ylabel('vertex index')

    # Format and save figure
    plt.tight_layout()
    plt.savefig(figname, dpi=200)

    ntnsubstatus('File name: '+str(figname))

# *******************************************************************

# Record generation
def get_record(voltage, spike_trains, graph, id):
    record = [
        id,
        average_pearson(voltage),
        average_pearson(spike_trains),
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
        bidirectional_edges(graph)
    ]
    return record

column_names = [
    'id',
    'voltage PC',
    'ST PC',
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
    'bidirectional edges'
]

def get_sim_id(fname, n):
        combinations = int(n*(n-1)/2)
        simulation_id = fname[-combinations-20:-20]
        return simulation_id

# ****************************************************************************#
# Plots
def pairplots(df, images_path, simulations_stem_prefix):
    sns_pairplot = sns.pairplot(df, kind='reg')
    sns_pairplot.savefig(images_path / (simulations_stem_prefix + 'pairplot_reg'))
    sns_pairplot = sns.pairplot(df)
    sns_pairplot.savefig(images_path / (simulations_stem_prefix + 'pairplot'))
    sns_pairplot = sns.pairplot(df, vars=[
                                    'voltage PC',
                                    'ST PC',
                                    'directional voltage PC',
                                    'directional ST PC',
                                    'directionality',
                                    'log maximal simplices',
                                    'bidirectional edges'
                                    ])
    sns_pairplot.savefig(images_path / (simulations_stem_prefix + 'pairplot_mini'))
    sns_pairplot = sns.pairplot(df, vars=[
                                    'voltage PC',
                                    'ST PC',
                                    'directional voltage PC',
                                    'directional ST PC',
                                    'directionality',
                                    'log maximal simplices',
                                    'bidirectional edges'
                                    ], kind = 'reg')
    sns_pairplot.savefig(images_path / (simulations_stem_prefix + 'pairplot_mini_reg'))

def boxplots(df, images_path, simulations_stem_prefix):
    figure = plt.figure(figsize=[10,6])
    ax = figure.add_subplot()
    sns.boxplot(data = df, x = 'bidirectional edges', y = 'voltage PC', ax = ax)
    figure.savefig(images_path / (simulations_stem_prefix + 'voltage_edges_bp'))
    figure = plt.figure(figsize=[10,6])
    ax = figure.add_subplot()
    sns.boxplot(data = df, x = 'maximal simplices', y = 'voltage PC', ax = ax)
    figure.savefig(images_path / (simulations_stem_prefix + 'voltage_msimp_bp'))
    figure = plt.figure(figsize=[10,6])
    ax = figure.add_subplot()
    sns.boxplot(data = df, x = 'bidirectional edges', y = 'directional voltage PC', ax = ax)
    figure.savefig(images_path / (simulations_stem_prefix + 'dvoltage_edges_bp'))
    figure = plt.figure(figsize=[10,6])
    ax = figure.add_subplot()
    sns.boxplot(data = df, x = 'maximal simplices', y = 'directional voltage PC', ax = ax)
    figure.savefig(images_path / (simulations_stem_prefix + 'dvoltage_msimp_bp'))
    columns = ['indegree range', 'outdegree range', 'bidegree range', 'degree range']
    titles = ['voltage_id_bp', 'voltage_od_bp', 'voltage_bd_bp', 'voltage_d_bp']
    for col, title in zip(columns,titles):
        figure = plt.figure(figsize=[10,6])
        ax = figure.add_subplot()
        sns.boxplot(data = df, x = col, y = 'voltage PC', ax = ax)
        figure.savefig(images_path / (simulations_stem_prefix + title))


def hueplots(df, images_path, simulations_stem_prefix):
    hues = [
        'indegree range',
        'outdegree range',
        'degree range',
        'directionality',
        'log maximal simplices'
    ]

    titles = [
        'voltage_bedge_idhue',
        'voltage_bedge_odhue',
        'voltage_bedge_dhue',
        'voltage_bedge_dirhue',
        'voltage_bedge_mshue'
   ]

    for hue, title in zip(hues, titles):
        figure = plt.figure(figsize=[8,6])
        ax = figure.add_subplot()
        sns.scatterplot(data = df, x = 'bidirectional edges', y = 'voltage PC', hue = hue, ax = ax, palette = 'Reds')
        figure.savefig(images_path /(simulations_stem_prefix + title))

    hues = [
        'indegree range',
        'outdegree range',
        'degree range',
        'directionality',
        'bidirectional edges'
    ]

    titles = [
        'voltage_msimp_idhue',
        'voltage_msimp_odhue',
        'voltage_msimp_dhue',
        'voltage_msimp_dirhue',
        'voltage_msimp_bedgeshue'
    ]

    for hue, title in zip(hues, titles):
        figure = plt.figure(figsize=[8,6])
        ax = figure.add_subplot()
        sns.scatterplot(data = df, x = 'log maximal simplices', y = 'voltage PC', hue = hue, ax = ax, palette = 'Reds')
        figure.savefig(images_path /(simulations_stem_prefix + title))


if __name__ == '__main__':
    #******************************************************************************#
    # Read arguments
    parser = argparse.ArgumentParser(
        description='n-simplex Circuit analysis tool',
        usage='python nsimplex-analyse.py')
    parser.add_argument('--root', type=str, default='.', help='Root directory for importing and exporting files')
    parser.add_argument('--save_path', type=str, default='3simplex/3simplex', help='Path to circuit simulation files, i.e. simulatons will be \'simulatons/{save_path}{id}-{data_type}\'.')
    parser.add_argument('--structure_path', type=str, default='3simplex/3simplex', help='Path to circuit exc matrices.')
    parser.add_argument('--n', type=int, default=3, help='Number of vertices in the graph.')
    parser.add_argument('--time', type=int, default=200, help='Length of the simulation')
    parser.add_argument('--binsize', type=int, default=5, help='Spike train bin size.')

    args = parser.parse_args()

    # Dataset generation and simple voltage traces
    simulations_root = Path(args.root + '/simulations/' + args.save_path).parent
    simulations_stem_prefix = Path(args.save_path).stem
    images_path = simulations_root / 'images'
    images_path.mkdir(exist_ok = True, parents = True)
    df = []
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
        if np.all([char == '0' for char in simulation_id]):
            plot_traces(volts, spikes, images_path / (simulations_stem_prefix + 'simple'))
        if np.all([char == '1' for char in simulation_id]):
            plot_traces(volts, spikes, images_path / (simulations_stem_prefix + 'full'))
        df.append(get_record(volt_array, spike_trains, graph, simulation_id))
    df = pd.DataFrame(df, columns = column_names)
    # Plot results
    pairplots(df, images_path, simulations_stem_prefix)
    boxplots(df, images_path, simulations_stem_prefix)
    hueplots(df, images_path, simulations_stem_prefix)
