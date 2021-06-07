# N-dimensional simplices analysis tool
# Date: April 2021

# Packages
import pandas as pd                                                         # For dataframe handling
import argparse                                                             # For options
from datetime import datetime                                               # For giving messages
from pathlib import Path                                                    # For file management

from utils.reliability_measures import gaussian_reliability, delayed_reliability
from utils.data import load_spikes, _spike_trains

# Formatted printer for messages
def ntnstatus(message):
    print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)


def ntnsubstatus(message):
    print('    '+message, flush=True)


# Reliability computation
def compute_reliabilities(path_df, _args):
    gaussian_sts = []
    delayed_sts = []
    for voltage_path in path_df['path']:
        spike_path = (voltage_path.replace('volts', 'spikes'))
        gaussian_sts.append(_spike_trains(load_spikes(spike_path), _args.n, _args.gaussian_binsize, _args.time))
        delayed_sts.append(_spike_trains(load_spikes(spike_path), _args.n, _args.gaussian_binsize, _args.time))
    return gaussian_reliability(gaussian_sts, _args.gaussian_variance), delayed_reliability(delayed_sts, _args.shift)


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser(
        description='n-simplex Circuit analysis tool',
        usage='python nsimplex-analyse.py')
    parser.add_argument('--root', type=str, default='.', help='Root directory for importing and exporting files')
    parser.add_argument('--save_path', type=str, default='3simplex/3simplex',
                        help='Path to circuit simulation output files.'
                             'All configurations simulations will be looked for'
                             ' as \'simulations/{save_path_parent}**dataframe.pkl\'.')
    parser.add_argument('--structure_path', type=str, default='3simplex/3simplex',
                        help='Path to circuit exc matrices. Default 3simplex/3simplex')
    parser.add_argument('--n', type=int, default=3, help='Number of vertices in the graph. Default 3.')
    parser.add_argument('--time', type=int, default=200, help='Length of the simulation. Default 200.')
    parser.add_argument('--delayed_binsize', type=int, default=5,
                        help='Spike train bin size for cross-correlation reliability. Default 5.')
    parser.add_argument('--gaussian_binsize', type=int, default=1,
                        help='Spike train bin size for gaussian reliability. Default 1.')
    parser.add_argument('--gaussian_variance', type=float, default=1,
                        help='Gaussian variance of filter. Default 1.')
    parser.add_argument('--shift', type=int, default=2,
                        help='Maximum shift for cross-correlation reliability. Default 2.')
    args = parser.parse_args()

    simulations_root = Path(args.root + '/simulations/' + args.save_path).parent
    simulations_stem_prefix = Path(args.save_path).stem
    images_path = simulations_root / 'images'
    images_path.mkdir(exist_ok=True, parents=True)
    # Plot results
    df = pd.concat([pd.read_pickle(df_path) for df_path in simulations_root.glob("**/*dataframe.pkl")
                    if "reliability_dataframe" not in str(df_path)], ignore_index=True)
    # Extra check to ensure no path conflict. Should be useless if save path of sims was properly set.
    # See the relevant bash file.
    df = df.drop_duplicates(subset='path')

    df = df.groupby('id').apply(compute_reliabilities, args)
    df.to_pickle(simulations_root /
                 (simulations_stem_prefix + "reliability_dataframe" +
                  "_gstd_"+str(args.gaussian_variance) +
                  "_shift_" + str(args.shift) + ".pkl"))
