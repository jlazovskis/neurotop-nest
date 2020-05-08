# Biological neural network model in NEST simulator. Based on BlueBrain neocortical microcircuit and experiments from Frontiers paper.
# Neuro-Topology Group, Institute of Mathematics, University of Aberdeen
# Authors: Jānis Lazovskis, Jason Smith
# Date: March 2020

# Packages
import numpy as np                                                          # For reading of files
import pandas as pd                                                         # For reading of files
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
	usage='python nest_mc2.py [--fibres=fibres.npy] [--stimulus=stim.npy] [--time=100] [--title="Plot title"]')

# Arguments: structure
parser.add_argument('--no_mc2approx', action='store_true', help='If not included, uses mc2 data for a better approximation.')
parser.add_argument('--shuffle', action='store_true', help='If included, randomly shuffles the mc2 adjacency matrix.')

# Arguments: stimulus
parser.add_argument('--fibres', type=str, default='fibres.npy', help='Filename of fibres in folder "stimuli". List of lists of GIDs. Length is number of thalamic fibers.')
parser.add_argument('--stimulus', type=str, default='constant_firing.npy', help='Filename of firing pattern of stimulus in folder "stimuli". List of tuples (index,start,stop,rate). Index has to be within range of length of fibres option.')
parser.add_argument('--time', type=int, default=100, help='Length, in milliseconds, of experiment. Must be an integer. Default is 100.')

# Arguments: outputs
parser.add_argument('--make_spikes', action='store_true', help='If included, outputs a file "spikes.h5" containing two lists, one of the GIDs of spiking neurons, the other of times at which they spiked. This is the dictionary NEST produces with getstatus for the events of the spikedetector.')
parser.add_argument('--make_tr', action='store_true', help='If included, outputs a file "tr.npz" containing transmission response matrices and "simplexcount.npy" containing simplex counts for each transmission response step.')
parser.add_argument('--plot_simplices', action='store_true', help='If included, outputs a plot simplexcount that shows the simplex count at each transimssion response step.')
parser.add_argument('--flagser', type=str, default='../flagser/flagser', help='Location of flagser executable.')
parser.add_argument('--t1', type=float, default=5.0, help='t1 for transmission reponse matrices')
parser.add_argument('--t2', type=float, default=10.0, help='t2 for transmission reponse matrices')
parser.add_argument('--no_plot', action='store_true', help='If not included, outputs visual plot of spikes and voltages of experiment.')
parser.add_argument('--outplottitle', type=str, default='Visual reports', help='Title for plot produced at the end. ')
parser.add_argument('--volt_plot', action='store_true', help='If included, output is only voltage plot.')
args = parser.parse_args()

# Set up
#nest.set_verbosity("M_ERROR")                                              # Uncomment this to make NEST quiet
nest.ResetKernel()                                                          # Reset nest
root = sys.argv[0][:-11]                                                    # Current working directory
class mc2simul:                                                             # Class for simulation
	id = datetime.now().strftime("%s")
	length = args.time

# Load circuit info
ntnstatus('Loading mc2 structural information')
nnum = 31346                                                                # Number of neurons in circuit
adj = np.load(root+'structure/adjmat_mc2.npz')                              # Adjacency matrix
mc2_edges = list(zip(adj['row'], adj['col']))                               # mc2 edges between neurons
mc2_excitatory = np.load(root+'structure/isneuronexcitatory_mc2.npy')       # Binary list indicating if neuron is excitatory or not
mc2_delays = np.load(root+'structure/distances_mc2.npz')['data']            # Interpret distances as delays
mc2_layers = np.load(root+'structure/layersize_mc2.npy')                    # Size of mc2 layers (for interpreting structural information)
mc2_layers_summed = [sum(mc2_layers[:i]) for i in range(55)]                # Summed layer sizes for easier access
mc2_weights = pd.read_pickle(root+'structure/synapses_mc2.pkl')             # Average number of synapses between layers
mc2_transmits = pd.read_pickle(root+'structure/failures_mc2.pkl')           # Average number of failures between layers

# Load stimulus info
fibres_address = root+'stimuli/'+args.fibres                                # Address of thalamic nerve fibres data
fibres = np.load(fibres_address,allow_pickle=True)                          # Get which neurons the fibres connect to
stimulus_address = root+'stimuli/'+args.stimulus
firing_pattern = np.load(stimulus_address,allow_pickle=True)                # When the nerve fibres fire, each element of the form (fibre_id,start_time,end_time,firing_rate)
stim_strength = 10000

# Declare parameters
weight_factor = 1.0                                                         # Weight of synapses (used as a multiplying factor)
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
	if not args.no_mc2approx:  
		layerpair = getlayers(mc2_edges[0][i],mc2_edges[1][i])
		nest.Connect((mc2_edges[0][i]+1,), (mc2_edges[1][i]+1,), syn_spec={
			'model' : 'bernoulli_synapse',
			'weight' : (0 if np.isnan(mc2_weights.iloc[layerpair]) else mc2_weights.iloc[layerpair])*weight_factor*(-1)**(not mc2_excitatory[mc2_edges[0][i]]),
			'delay' : mc2_delays[i]*delay_factor,
			'p_transmit' : 1-(max_fail if np.isnan(mc2_transmits.iloc[layerpair]) else mc2_transmits.iloc[layerpair])/max_fail})
	else:
		nest.Connect((mc2_edges[0][i]+1,), (mc2_edges[1][i]+1,), syn_spec={'weight': np.random.random()*weight_factor, 'delay': delay_factor})

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
	 'withgid': True})
spikedetector = nest.Create('spike_detector', params={
	'label': 'spikes',
	'withgid': True})
for n in range(1,nnum+1):
	nest.Connect(voltmeter,(n,))
	nest.Connect((n,),spikedetector)

# Add ambient noise
ntnstatus('Creating ambient noise for circuit')
noisers = nest.Create('noise_generator', n=nnum)
for i in range(nnum):
	nest.SetStatus((noisers[i],), params={'mean': 150+np.random.rand()*50, 'std':np.random.rand()*20, 'dt':0.1*np.random.randint(1,10), 'phase':np.random.rand()*360, 'frequency':np.random.rand()*1000000})
	nest.Connect((noisers[i],), (i+1,))

# Run simulation
ntnstatus("Running simulation of "+str(exp_length)+"ms")
nest.Simulate(float(exp_length))

# Process results
from nest_mc2_output import *
cur_simul = mc2simul()
cur_simul.neurons = nnum
cur_simul.stimulus = firing_pattern
cur_simul.voltage = nest.GetStatus(voltmeter)[0]['events']['V_m']
cur_simul.spikes = nest.GetStatus(spikedetector)[0]['events']

if args.make_spikes:
	ntnstatus("Creating h5 file of spikes")
	make_spikes(cur_simul)

if args.make_tr:
	t1 = args.t1
	t2 = args.t2
	ntnstatus("Creating transmission response matrices with t1="+str(t1)+" and t2="+str(t2))
	cur_simul.adj = adj
	cur_simul.simplices = make_tr(cur_simul, t1, t2, args.flagser)

if not args.no_plot:
	if args.volt_plot:
		ntnstatus("Creating voltage plot")
		make_volt_plot(cur_simul)
	else:
		ntnstatus("Creating spike and volt plots")
		make_plot(cur_simul, args.outplottitle, args.plot_simplices)