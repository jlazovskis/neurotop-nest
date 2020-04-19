# Biological neural network model in NEST simulator. Based on BlueBrain neocortical microcircuit and experiments from Frontiers paper.
# Neuro-Topology Group, Institute of Mathematics, University of Aberdeen
# Authors: Jānis Lazovskis, Jason Smith
# Date: March 2020

# Packages: for inputs
import numpy as np                                                          # For reading of files
import pandas as pd                                                         # For reading of files
import random                                                               # For list shuffling
import sys                                                                  # For current directory to read and write files
import nest                                                                 # For main simulation
import argparse                                                             # For options

# Packages: for outputs
from datetime import datetime                                               # For giving messages
import h5py                                                                 # For exporting h5 files
import matplotlib as mpl                                                    # For plotting
import matplotlib.pyplot as plt                                             # For plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable                     # For plotting, to add colorbar nicely

# Auxiliary: Formatted printer for messages
def ntnstatus(message):
	print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)

# Auxiliary: Get layer pairs from neuron pairs
def getlayers(source,target):
	source_layer = 55 - [int(lay/(source+1)) for lay in reversed(mc2_layers_summed)].index(0)
	target_layer = 55 - [int(lay/(target+1)) for lay in reversed(mc2_layers_summed)].index(0)
	return (source_layer-1,target_layer-1)

# Read arguments
parser = argparse.ArgumentParser(
	description='Simplified Blue Brain Project reconstructions and validations',
	usage='python nest_mc2.py [--fibres=fibres.npy] [--stimulus=stim.npy] [--time=100] [--title="Plot title"]')

# Arguments: structure
parser.add_argument('--no_mc2approx', action='store_false', help='If not included, uses mc2 data for a better approximation.')
parser.add_argument('--shuffle', action='store_true', help='If included, randomly shuffles the mc2 adjacency matrix.')

# Arguments: stimulus
parser.add_argument('--fibres', type=str, default='fibres.npy', help='Filename of fibres in folder "stimuli". List of lists of GIDs. Length is number of thalamic fibers.')
parser.add_argument('--stimulus', type=str, default='constant_firing.npy', help='Filename of firing pattern of stimulus in folder "stimuli". List of tuples (index,start,stop,rate). Index has to be within range of length of fibres option.')
parser.add_argument('--time', type=int, default=100, help='Length, in milliseconds, of experiment. Must be an integer. Default is 100.')

# Arguments: outputs
parser.add_argument('--outspikes', action='store_true', help='If included, outputs a file "spikes.h5" containing two lists, one of the GIDs of spiking neurons, the other of times at which they spiked. This is the dictionary NEST produces with getstatus for the events of the spikedetector.')
parser.add_argument('--no_outplot', action='store_false', help='If not included, outputs visual plot of spikes and voltages of experiment.')
parser.add_argument('--outplottitle', type=str, default='Spikemeter and voltmeter reports', help='Title for plot produced at the end. ')
args = parser.parse_args()

# Set up
#nest.set_verbosity("M_ERROR")                                              # Uncomment this to make NEST quiet
nest.ResetKernel()                                                          # Reset nest
root = sys.argv[0][:-11]                                                    # Current working directory
										
# Load circuit info
ntnstatus('Loading mc2 structural information')
mc2_address = root+'structure/adjmat_mc2.npz'                               # mc2 adjacency matrix
adj = np.load(mc2_address)
mc2_edges = list(zip(adj['row'], adj['col']))                               # mc2 edges between neurons
distance_address = root+'structure/distances_mc2.npz'                       # mc2 physical distances between neurons
mc2_delays = np.load(distance_address)['data']                              # Interpret distances as delays
mc2_layers = np.load(root+'structure/layersize_mc2.npy')                    # Size of mc2 layers (for interpreting structural information)
mc2_layers_summed = [sum(mc2_layers[:i]) for i in range(55)]                # Summed layer sizes for easier access
mc2_weights = pd.read_pickle(root+'structure/synapses_mc2.pkl')             # Average number of synapses between layers
mc2_transmits = pd.read_pickle(root+'structure/failures_mc2.pkl')           # Average number of failures between layers
nnum = 31346                                                                # Number of neurons in circuit

# Load stimulus info
fibres_address = root+'stimuli/'+args.fibres                                # Address of thalamic nerve fibres data
fibres = np.load(fibres_address,allow_pickle=True)                          # Get which neurons the fibres connect to
stimulus_address = root+'stimuli/'+args.stimulus
firing_pattern = np.load(stimulus_address,allow_pickle=True)                # When the nerve fibres fire, each element of the form (fibre_id,start_time,end_time,firing_rate)
stim_strength = 2000000

# Declare parameters
augold2 = (1., 0.866667, 0.)                                                # Declare color scheme
weight_factor = 10.0                                                        # Weight of synapses (used as a multiplying factor)
delay_factor = 0.1                                                          # Delay between neurons (used as a multiplying factor)
exp_length = args.time                                                      # Length of experiment, in milliseconds
max_fail = 100                                                              # Number of failures to consider as no transmission for a synapse

# Shuffle the neuron connections
if args.shuffle:
	edge_num = len(mc2_edges)
	mc2_edges = list(zip(random.sample(list(adj['row']), edge_num), random.sample(list(adj['col']), edge_num)))

# Create the circuit
ntnstatus('Constructing circuit')
network = nest.Create('iaf_cond_exp_sfa_rr', n=nnum, params=None)
for i in range(len(mc2_edges[0])):
	if args.mc2approx:  
		layerpair = getlayers(mc2_edges[0][i],mc2_edges[1][i])
		nest.Connect((mc2_edges[0][i]+1,),(mc2_edges[1][i]+1,), syn_spec={
			'weight' : (0 if np.isnan(mc2_weights.iloc[layerpair]) else mc2_weights.iloc[layerpair])*weight_factor,
			'delay' : mc2_delays[i]*delay_factor,
			'model' : 'bernoulli_synapse',
			'p_transmit' : 1-(max_fail if np.isnan(mc2_transmits.iloc[layerpair]) else mc2_transmits.iloc[layerpair])/max_fail})
	else:
		nest.Connect((mc2_edges[0][i]+1,),(mc2_edges[1][i]+1,), syn_spec={'weight': np.random.random()*weight_factor, 'delay': delay_factor})

# Define stimulus and connect it to neurons
ntnstatus('Creating thalamic nerves for stimulus')
stimuli = nest.Create('poisson_generator', n=len(fibres))
for stimulus in range(len(fibres)):
	for j in fibres[stimulus]:
		nest.Connect((stimuli[stimulus],),(j+1,))

# Record voltage and spikes
ntnstatus("Connecting thalamic nerves to circuit")
voltmeter = nest.Create('voltmeter', params={
	 'label': 'volts',
	 'withtime': True,
	 'withgid': True})
spikedetector = nest.Create('spike_detector', params={
	'label': 'spikes',
	'withgid': True})
for n in range(1,nnum+1):
	nest.Connect(voltmeter,(n,))
	nest.Connect((n,),spikedetector)
#    nest.SetStatus((n,), {"I_e": 250.0+np.random.rand()*stim*.5})

for fire in firing_pattern:
	nest.SetStatus((stimuli[int(fire[0])],), params={
		 'start':float(fire[1]),
		 'stop':float(fire[2]),
		 'rate': float(fire[3])*stim_strength})

# Run simulation
ntnstatus("Running simulation of "+str(exp_length)+"ms")
nest.Simulate(float(exp_length))

# Make reports of experiment
timestamp = datetime.now().strftime("%s")
if args.outspikes:
	ntnstatus("Creating h5 file of spikes")
	f = h5py.File('spikes_'+timestamp+'.h5','w')
	spikes = nest.GetStatus(spikedetector)[0]['events']
	for k in spikes.keys():
		f.create_dataset(k, data=spikes[k])
	f.close()

if args.no_outplot:
	ntnstatus("Creating spike and volt plots")
	spikes = nest.GetStatus(spikedetector)[0]['events']
	volts = nest.GetStatus(voltmeter)[0]['events']['V_m']
	fig = plt.figure(figsize=(15,5)) # default is (8,6)
	fig.suptitle(args.outplottitle,fontsize=18)
	
	ax_spikes = fig.add_subplot(2,1,1)
	ax_spikes.scatter(spikes['times'], spikes['senders'], s=1, marker="+")
	ax_spikes.set_ylabel('neuron index')
	ax_spikes.set_xticks([])
	ax_spikes.set_ylim(1,nnum)
	plt.gca().invert_yaxis()
	ax_spikes.set_xlim(1,exp_length)
	
	ax_volts = fig.add_subplot(212)
	v = ax_volts.imshow(np.transpose(np.array(volts).reshape(int(exp_length-1),nnum)), cmap=plt.cm.Spectral_r, interpolation='None', aspect="auto")
	ax_volts.set_ylabel('neuron index')
	ax_volts.set_xlabel('time in ms')
	
	fig.colorbar(v, ax=[ax_spikes,ax_volts], pad=0.03, fraction=0.02, orientation='vertical', label="voltage")
	plt.savefig(root+'report_'+timestamp+'.png')
