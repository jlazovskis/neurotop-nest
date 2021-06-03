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

from tqdm import tqdm

from utils.reliability_measures import gaussian_reliability, delayed_reliability
from nsimplex_analyse import _spike_trains, load_spikes
#******************************************************************************#
# Formatted printer for messages
def ntnstatus(message):
    print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)
def ntnsubstatus(message):
    print('    '+message, flush=True)





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
    parser.add_argument('--delayed_binsize', type=int, default=5, help='Spike train bin size for delayed reliability.')
    parser.add_argument('--gaussian_binsize', type=int, default=1, help='Spike train bin size for gaussian reliability.')
    parser.add_argument('--gaussian_variance', type=float, default=1, help='Gaussian variance of filter')
    parser.add_argument('--shift', type=int, default=2, help='Shift for delayed reliability')
    args = parser.parse_args()

    # Sample voltage traces
    simulations_root = Path(args.root + '/simulations/' + args.save_path).parent
    simulations_stem_prefix = Path(args.save_path).stem
    images_path = simulations_root / 'images'
    images_path.mkdir(exist_ok = True, parents = True)
    # Plot results
    df = pd.concat([pd.read_pickle(df_path) for df_path in simulations_root.glob("**/*dataframe.pkl") if "reliability_dataframe" not in str(df_path)], ignore_index=True)
    df = df.drop_duplicates(subset = 'path')
    def compute_reliabilities(path_df):
        gaussian_sts = []
        delayed_sts = []
        for voltage_path in path_df['path']:
            spike_path = (voltage_path.replace('volts', 'spikes'))
            gaussian_sts.append(_spike_trains(load_spikes(spike_path), args.n, args.gaussian_binsize, args.time))
            delayed_sts.append(_spike_trains(load_spikes(spike_path), args.n, args.gaussian_binsize, args.time))
        return gaussian_reliability(gaussian_sts, args.gaussian_variance), delayed_reliability(delayed_sts, args.shift)
    df = df.groupby('id').apply(compute_reliabilities)
    df.to_pickle(simulations_root / (simulations_stem_prefix + "reliability_dataframe_gstd"+str(args.gaussian_variance) + "shift" + str(args.shift) + ".pkl"))
