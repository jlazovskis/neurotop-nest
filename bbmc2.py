# neurotop-nest
#
# Based on BlueBrain neocortical microcircuit
# https://bbp.epfl.ch/nmc-portal/web/guest/downloads#
# Neuro-Topology Group, Institute of Mathematics, University of Aberdeen
# Authors: Jānis Lazovskis, Jason Smith
# Date: March 2020

# Packages
import numpy as np                                                          # For reading of files
import pandas as pd                                                         # For reading of files
from scipy.sparse import load_npz                                           # For reading of files
import random                                                               # For list shuffling
import sys                                                                  # For current directory to read and write files
import nest                                                                 # For main simulation
import argparse                                                             # For options
from datetime import datetime                                               # For giving messages

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
	usage='python nest_mc2.py [--fibres=fibres.npy] [--stimulus=stim.npy] [--time=100]')
parser.add_argument('--stimulus', type=str, default='constant', help='Firing pattern to use. One of "n5", "n15", "n30", "constant". File is a list of tuples (index,start,stop,rate). Index has to be within range of length of fibres option.')
parser.add_argument('--fibres', type=str, default='fibres.npy', help='Filename of fibres in folder "stimuli". List of lists of GIDs. Length is number of thalamic fibers.')
parser.add_argument('--time', type=int, default=100, help='Length, in milliseconds, of experiment. Must be an integer. Default is 100.')
parser.add_argument('--no_mc2approx', action='store_true', help='If not included, uses mc2 data for a better approximation.')
parser.add_argument('--shuffle', action='store_true', help='If included, randomly shuffles the mc2 adjacency matrix.')
args = parser.parse_args()

# Set up
#nest.set_verbosity("M_ERROR")                                              # Uncomment this to make NEST quiet
nest.ResetKernel()                                                          # Reset nest
nest.SetKernelStatus({"local_num_threads": 8})                              # Run on many threads
root = sys.argv[0][:-11]                                                    # Current working directory
simulation_id = datetime.now().strftime("%s")                               # Custom ID so files are not overwritten

# Load circuit info
ntnstatus('Loading mc2 structural information')
nnum = 31346                                                                # Number of neurons in circuit
adj = load_npz(root+'structure/adjmat_mc2.npz').toarray()                   # Adjacency matrix
exc = np.load(root+'structure/isneuronexcitatory_mc2.npy')                  # Binary list indicating if neuron is excitatory or not
mc2_delays = np.load(root+'structure/distances_mc2.npz')['data']            # Interpret distances as delays
mc2_layers = np.load(root+'structure/layersize_mc2.npy')                    # Size of mc2 layers (for interpreting structural information)
mc2_layers_summed = [sum(mc2_layers[:i]) for i in range(55)]                # Summed layer sizes for easier access
mc2_weights = pd.read_pickle(root+'structure/synapses_mc2.pkl')             # Average number of synapses between layers
mc2_transmits = pd.read_pickle(root+'structure/failures_mc2.pkl')           # Average number of failures between layers

# Load stimulus info
fibres_address = root+'stimuli/'+args.fibres                                # Address of thalamic nerve fibres data
fibres = np.load(fibres_address,allow_pickle=True)                          # Get which neurons the fibres connect to
stim_dict = {
	'n5':'stim5_firing_pattern.npy',
	'n15':'stim15_firing_pattern.npy',
	'n30':'stim30_firing_pattern.npy',
	'constant':'constant_firing.npy'}
firing_pattern = np.load(root+'stimuli/'+stim_dict[args.stimulus],allow_pickle=True)
stim_strength = 50

# Declare parameters
syn_weight = 1.0                                                            # Weight of excitatory synapses
inh_fac = -1.0                                                              # Multiplier weight of inhibitory synapses
delay = 0.1                                                                 # Delay between neurons (used as a multiplying factor)
exp_length = args.time                                                      # Length of experiment, in milliseconds
max_fail = 100                                                              # Number of failures to consider as no transmission for a synapse

# Shuffle the neuron connections
if args.shuffle:
	edge_num = len(mc2_edges)
	mc2_edges = list(zip(random.sample(list(adj['row']), edge_num), random.sample(list(adj['col']), edge_num)))

# Create the circuit
ntnstatus('Constructing circuit')
network = nest.Create('izhikevich', n=nnum, params={'a':1.1})
#network = nest.Create('iaf_cond_exp_sfa_rr', n=nnum, params=None)
targets = {n:np.nonzero(adj[n])[0] for n in range(nnum)}
for source in targets.keys():
	nest.Connect((source+1,), [target+1 for target in targets[source]], conn_spec='all_to_all', syn_spec={
		'model':'bernoulli_synapse',
		'weight':syn_weight if exc[source] else inh_fac*syn_weight,
		'delay':delay,
		'p_transmit':0.1})
#	if not args.no_mc2approx:  
#		layerpair = getlayers(mc2_edges[0][i],mc2_edges[1][i])
#		nest.Connect((mc2_edges[0][i]+1,), (mc2_edges[1][i]+1,), syn_spec={
#			'model' : 'bernoulli_synapse',
#			'weight' : (0 if np.isnan(mc2_weights.iloc[layerpair]) else mc2_weights.iloc[layerpair])*weight_factor*(-1)**(not mc2_excitatory[mc2_edges[0][i]]),
#			'delay' : mc2_delays[i]*delay_factor,
#			'p_transmit' : 1-(max_fail if np.isnan(mc2_transmits.iloc[layerpair]) else mc2_transmits.iloc[layerpair])/max_fail})
#	else:
#		nest.Connect((mc2_edges[0][i]+1,), (mc2_edges[1][i]+1,), syn_spec={'weight': np.random.random()*weight_factor, 'delay': delay_factor})

# Define stimulus and connect it to neurons
ntnstatus('Creating thalamic nerves for stimulus')
stimuli = nest.Create('poisson_generator', n=len(fibres))
for stimulus in range(len(fibres)):
	for j in fibres[stimulus]:
		nest.Connect((stimuli[stimulus],),(j+1,))
for fire in firing_pattern:
	nest.SetStatus((stimuli[int(fire[0])],), params={
		 'start': round(float(fire[1]),1),
		 'stop': round(float(fire[2]),1),
		 'rate': float(fire[3])*stim_strength})

# Connect voltage and spike readers
ntnstatus('Adding voltage and spike readers')
voltmeter = nest.Create('voltmeter', params={
	 'label': 'volts',
	 'withtime': True,
	 'withgid': True,
	 'interval': 0.1})
spikedetector = nest.Create('spike_detector', params={
	'label': 'spikes',
	'withgid': True})
for n in range(1,nnum+1):
	nest.Connect(voltmeter,(n,))
	nest.Connect((n,),spikedetector)

# Add ambient noise
ntnstatus('Creating ambient noise for circuit')
weak_noise = nest.Create('noise_generator', params={'mean':1.5, 'std':0.5})
nest.Connect(weak_noise, list(range(1,31347)), conn_spec='all_to_all')
strong_num = 1500
strong_perc = 0.015
strong_times = np.random.choice([i/10 for i in range(2,exp_length*10)], size=strong_num)
for i in range(strong_num):
	strong_targets = list(np.random.choice(range(1,31347),size=int(nnum*strong_perc),replace=False))
	strong_noise = nest.Create('noise_generator', params={'mean':30.0, 'std':0.5, 'start':strong_times[i]-.1, 'stop':strong_times[i]+.1})
	nest.Connect(strong_noise, strong_targets, conn_spec='all_to_all')

# # Add spontaneous spikes
# ntnstatus('Creating spontaneous spikes for circuit')
# spontaneous_times = np.random.choice([i/10 for i in range(1,exp_length*10)], size=500)
# spontaneous_times.sort()
# spontaneous_generator = nest.Create('spike_generator', params={'spike_times':spontaneous_times})
# spontaneous_spreader = nest.Create('spike_dilutor', params={'p_copy':1/nnum})
# nest.Connect(spontaneous_generator,spontaneous_spreader)
# nest.Connect(spontaneous_generator,list(range(1,31347)), conn_spec='all_to_all')

# Run simulation
ntnstatus("Running simulation of "+str(exp_length)+"ms")
nest.Simulate(float(exp_length))
#v = nest.GetStatus(voltmeter)[0]['events']['V_m']     # volts
s = nest.GetStatus(spikedetector)[0]['events']        # spikes

# Save results
np.save(root+'bbmc2_'+str(args.stimulus)+'_'+simulation_id+'.npy', np.array(s))
