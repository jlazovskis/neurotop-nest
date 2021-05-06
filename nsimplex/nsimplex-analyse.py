# A model using n-dimensional simplices
# Date: April 2021

# Packages
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import sys                                                                  # For current directory to read and write files
import argparse                                                             # For options
from datetime import datetime                                               # For giving messages
import random
import matplotlib.pyplot as plt                                             # For plotting
import matplotlib as mpl                                                    # For plotting

#******************************************************************************#
# Read arguments
parser = argparse.ArgumentParser(
    description='n-simplex Nest Circuit',
    usage='python nsimplex.py')
parser.add_argument('--root', type=str, default='.', help='Root directory for importing and exporting files')
parser.add_argument('--circuit', type=str, default='3simplex', help='Path to circuit file, without .npy.')

parser.add_argument('--simulation_id', type=int, default=0, help='Simulation id')
parser.add_argument('--number_vertices', type=int, default=4, help='Number of vertices in the graph.')

parser.add_argument('--time', type=int, default=100, help='Length of the simulation')

args = parser.parse_args()

#******************************************************************************#
# Formatted printer for messages
def ntnstatus(message):
    print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)
def ntnsubstatus(message):
    print('    '+message, flush=True)


#******************************************************************************#
# Configuration
simulation_id = str(args.simulation_id)
ntnstatus('Loading results of simulation '+simulation_id)
root = args.root
nnum = args.number_vertices
volts_array = np.load(root+'/simulations/'+str(args.circuit)+'_'+simulation_id+'-volts.npy', allow_pickle=True)
volts = {'senders':volts_array[0], 'times':volts_array[1], 'V_m':volts_array[2]}

spikes_array = np.load(root+'/simulations/'+str(args.circuit)+'_'+simulation_id+'-spikes.npy', allow_pickle=True)
spikes = {'senders':spikes_array[0], 'times':spikes_array[1]}

#******************************************************************************#
# Plot the results
ntnstatus("Plotting results")

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
plt.savefig(root+'/simulations/'+simulation_id+'.png', dpi=200)

ntnsubstatus('File name: '+simulation_id+'.png')
