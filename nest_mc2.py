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
from functools import reduce                                                # For combining simplices
import matplotlib as mpl                                                    # For plotting
import matplotlib.pyplot as plt                                             # For plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable                     # For plotting, to add colorbar nicely
import scipy.sparse                                                         # For exporting transimssion response matrices
import subprocess                                                           # For counting simplices and running flagser

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
parser.add_argument('--make_tr', action='store_true', help='If included, outputs a file "tr.npz" containing transmission response matrices.')
parser.add_argument('--count_simplices', action='store_true', help='If included, outputs a file "simplexcount.npy" containing simplex counts for each transmission response step.')
parser.add_argument('--flagser', type=str, default='../flagser/flagser', help='Location of flagser executable.')
parser.add_argument('--t1', type=float, default=5.0, help='t1 for transmission reponse matrices')
parser.add_argument('--t2', type=float, default=10.0, help='t2 for transmission reponse matrices')
parser.add_argument('--no_plot', action='store_true', help='If not included, outputs visual plot of spikes and voltages of experiment.')
parser.add_argument('--outplottitle', type=str, default='Visual reports', help='Title for plot produced at the end. ')
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
nnum = 31346                                                                # Number of neurons in circuit
if not args.no_mc2approx:
	distance_address = root+'structure/distances_mc2.npz'                   # mc2 physical distances between neurons
	mc2_delays = np.load(distance_address)['data']                          # Interpret distances as delays
	mc2_layers = np.load(root+'structure/layersize_mc2.npy')                # Size of mc2 layers (for interpreting structural information)
	mc2_layers_summed = [sum(mc2_layers[:i]) for i in range(55)]            # Summed layer sizes for easier access
	mc2_weights = pd.read_pickle(root+'structure/synapses_mc2.pkl')         # Average number of synapses between layers
	mc2_transmits = pd.read_pickle(root+'structure/failures_mc2.pkl')       # Average number of failures between layers

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
	if not args.no_mc2approx:  
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

# Report: spiketrains
timestamp = datetime.now().strftime("%s")
if args.make_spikes:
	ntnstatus("Creating h5 file of spikes")
	f = h5py.File('spikes_'+timestamp+'.h5','w')
	spikes = nest.GetStatus(spikedetector)[0]['events']
	for k in spikes.keys():
		f.create_dataset(k, data=spikes[k])
	f.close()

# Report: transmission response matrices and simplex count
if args.make_tr or args.count_simplices:
	t1 = args.t1
	t2 = args.t2
	ntnstatus("Creating transmission response matrices with t1="+str(t1)+" and t2="+str(t2))
	# Get key times
	times = [t1*i for i in range(int(exp_length/t1)+1)]
	# Get ordered times and indices of spikes
	spikes = nest.GetStatus(spikedetector)[0]['events']
	tr_times = sorted(spikes['times'])
	tr_neurons = [x-1 for _,x in sorted(zip(spikes['times'], spikes['senders']))]
	# Get adjacencies as dense matrix
	adjmat = scipy.sparse.coo_matrix((adj['data'], (adj['row'],adj['col'])), shape=(nnum,nnum)).toarray()
	matrices = []
	simplices = []
	for i in range(len(times)-1):
		print('    Step '+str(i)+': ['+str(times[i])+','+str(times[i+1])+'] in ['+str(times[i])+','+str(min(times[i]+t2,exp_length))+']',flush=True)
		M = np.zeros((nnum,nnum),dtype='int8')
		t1_start = np.searchsorted(tr_times, times[i])
		t1_end = np.searchsorted(tr_times, times[i+1])
		t2_end = np.searchsorted(tr_times, times[i]+t2)
		# Source vertex spiked in [0, t1], sink in [0,t2]
		sources = np.unique(tr_neurons[t1_start:t1_end])
		targets = np.unique(tr_neurons[t1_start:t2_end])
		target_vector = scipy.sparse.coo_matrix((np.ones(len(targets),dtype='int8'), (np.zeros(len(targets),dtype='int8'), targets)), shape=(1,nnum+1)).toarray()[0][1:]
		for source in sources:
			M[source] = np.logical_and(adjmat[source],target_vector)
		# Source vertex spiked in [t1,t2], sink in [t1,t2]
		sources = np.unique(tr_neurons[t1_end:t2_end])
		target_vector = scipy.sparse.coo_matrix((np.ones(len(sources),dtype='int8'), (np.zeros(len(sources),dtype='int8'), sources)), shape=(1,nnum+1)).toarray()[0][1:]
		for source in sources:
			M[source] = np.logical_and(adjmat[source],target_vector)
		matrices.append(M)
		if args.count_simplices:
			f = open(root+'step'+str(i),'w')
			f.write('dim 0\n')
			for j in range(nnum):
				f.write('0 ')
			f.write('\ndim 1\n')
			for j in range(nnum):
				for k in np.nonzero(matrices[i][j])[0]:
					f.write(str(j)+' '+str(k)+'\n')
			f.close()
			cmd = subprocess.Popen(['./'+str(args.flagser), '--out', root+'step'+str(i)+'.flag',  root+'step'+str(i)], stdout=subprocess.DEVNULL)
			cmd.wait()
			g = open(root+'step'+str(i)+'.flag','r')
			L = g.readlines()
			g.close()
			simplices.append(np.array(list(map(lambda x: int(x), L[1][:-1].split(' ')[1:]))))
			print('    Simplex counts > 1: '+reduce((lambda x,y: str(x)+' '+str(y)), simplices[-1]), flush=True)
			subprocess.Popen(['rm', root+'step'+str(i)], stdout=subprocess.DEVNULL)
			subprocess.Popen(['rm', root+'step'+str(i)+'.flag'], stdout=subprocess.DEVNULL)
	if args.make_tr:
		ntnstatus('Compressing '+str(len(matrices))+' dense matrices into npz file')
		np.savez_compressed(root+'transmissionresponse_'+timestamp, *matrices)
	if args.count_simplices:
		maxlen = max([len(s) for s in simplices])
		for s in range(len(simplices)):
			while len(simplices[s]) < maxlen:
				simplices[s] = np.concatenate((simplices[s],np.zeros(1,dtype='int64')), axis=None)
		np.save(root+'simplices_'+timestamp, np.array(simplices))

# Report: spike and voltage plots
if not args.no_plot:
	ntnstatus("Creating spike and volt plots")
	spikes = nest.GetStatus(spikedetector)[0]['events']
	volts = nest.GetStatus(voltmeter)[0]['events']['V_m']
	fig = plt.figure(figsize=(15,6)) # default is (8,6)
	fig.suptitle(args.outplottitle,fontsize=18)
	
	if args.count_simplices:
		ax_spikes = fig.add_subplot(3,1,1)
		ax_volts = fig.add_subplot(3,1,2)
		ax_simp = fig.add_subplot(3,1,3); plt.yscale("symlog")
		for dim in range(len(simplices[0])-1):
			ax_simp.plot(range(len(simplices)), [simplices[step][dim] for step in range(len(simplices))], label='dim'+str(dim+1))
		ax_simp.legend(bbox_to_anchor=(1, .5), loc='center left', borderaxespad=0.5)
		ax_simp.set_xticks(list(range(len(simplices))))
		ax_simp.set_ylabel('number of simplices')

	else:
		ax_spikes = fig.add_subplot(2,1,1)
		ax_volts = fig.add_subplot(2,1,2)

	ax_spikes.invert_yaxis()
	ax_spikes.scatter(spikes['times'], spikes['senders'], s=1, marker="+")
	ax_spikes.set_ylabel('neuron index')
	ax_spikes.set_ylim(1,nnum)
	ax_spikes.set_xlim(0,exp_length-1)
	
	v = ax_volts.imshow(np.transpose(np.array(volts).reshape(int(exp_length-1),nnum)), cmap=plt.cm.Spectral_r, interpolation='None', aspect="auto")
	ax_volts.invert_yaxis()
	ax_volts.set_ylabel('neuron index')
	ax_volts.set_xlabel('time in ms')
	
	fig.colorbar(v, ax=ax_volts, pad=0.03, fraction=0.02, orientation='vertical', label="voltage")
	plt.savefig(root+'report_'+timestamp+'.png')
