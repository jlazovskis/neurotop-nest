# Based on Janelia's drosophila connectome model
# http://neuprint.janelia.org/
# https://www.biorxiv.org/content/10.1101/2020.05.18.102061v1
# Neuro-Topology Group, Institute of Mathematics, University of Aberdeen
# Authors: Jānis Lazovskis, Jason Smith
# Date: June 2020

# Packages
import sys                                                                  # For current directory to read and write files
import nest                                                                 # For main simulation
import argparse                                                             # For options
from datetime import datetime                                               # For giving messages
from scipy.sparse import load_npz                                           # For reading of files
import numpy as np                                                          # For reading of files
import random
import tqdm                                                                 # For visualizing progress bar
from functools import reduce

# Auxiliary: Formatted printer for messages
def ntnstatus(message):
	print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)
def ntnsubstatus(message):
	print('    '+message, flush=True)

# Read arguments
parser = argparse.ArgumentParser(
	description='Drosophila reconstruction and validations',
	usage='python drosophila.py')
parser.add_argument('--time', type=int, default=100, help='Length, in milliseconds, of experiment. Must be an integer. Default is 100.')
args = parser.parse_args()

# Set up
#nest.set_verbosity("M_ERROR")                                              # Uncomment this to make NEST quiet
nest.ResetKernel()                                                          # Reset nest
nest.SetKernelStatus({"local_num_threads": 8})                              # Run on many threads
root = ''                                                                   # Current working directory
simulation_id = datetime.now().strftime("%s")                               # Custom ID so files are not overwritten

# Load circuit info
ntnstatus('Loading drosophila structural information')
nnum = 21663                                                                                # Number of neurons in circuit
snum = 13603750                                                                             # Number of synapses in circuit, computed by sum(sum(adj))
adj = load_npz(root+'structure/weighted_adjmat_drosophila.npz').toarray().astype('int64')   # Adjacency matrix

# Inhibitory connections                                                                                   # Uncomment next four lines to select new inhibitory connections
inh_prop = 0.1                                                                                             # Proportion of connections which are inhibitory
# edges=np.where(adj)                                                                                      # Get all connections
# inh=[(edges[0][i],edges[1][i]) for i in random.sample(range(len(edges[0])),int(len(edges[0])*inh_prop))] # randomly select some to be inhibitory
# np.save(root+'structure/drosophila_inhibitory.npy',inh)
# inh = np.load(root+'structure/drosophila_inhibitory.npy')                                                # list of pairs (X,Y) indicating X->Y is an inhibitory edge
inh_syn_integer = np.random.choice(range(snum),size=int(inh_prop*snum),replace=False)                      # Randomly select synapses to be inhibitory
inh_syn_binary = np.zeros(snum,dtype='int64')
for n in inh_syn_integer:
	inh_syn_binary[n] = 1

# Declare parameters
exc = {'weight':0.3, 'delay':1.0, 'transmit':0.13}                         # Excitatory synapse parameters
inh = {'weight':-1.5, 'delay':0.1, 'transmit':0.13}                         # Inhibitory synapse parameters
exp_length = args.time                                                      # Length of experiment, in milliseconds
stim_strength = 1000
noise_strength = 2.5

# Create the circuit
ntnstatus('Constructing circuit')
network = nest.Create('izhikevich', n=nnum, params={'a':1.1})
synapse_index = 0
for source in tqdm.tqdm(range(nnum)):
	targets = adj[source]
	out_neighbours = np.nonzero(targets)[0]
	out_degree = sum(targets)
	if out_degree:
		syn_integer = np.concatenate([np.array([target+1]*targets[target]) for target in out_neighbours]).ravel()
		syn_binary = inh_syn_binary[synapse_index:synapse_index+out_degree]
		nest.Connect((source+1,), syn_integer[np.where(syn_binary == 0)[0]].tolist(), conn_spec='all_to_all', syn_spec={
			'model': 'bernoulli_synapse',
			'weight': exc['weight'],
			'delay': exc['delay'],
			'p_transmit': exc['transmit']})
		nest.Connect((source+1,), syn_integer[np.where(syn_binary == 1)[0]].tolist(), conn_spec='all_to_all', syn_spec={
			'model': 'bernoulli_synapse',
			'weight': inh['weight'],
			'delay': inh['delay'],
			'p_transmit': inh['transmit']})
		synapse_index += out_degree

# Load stimulus and connect it to neurons
ntnstatus('Creating thalamic nerves for stimulus')
fibres = np.load(root+'stimuli/drosophila_optic_fibres.npy', allow_pickle=True)
firing_pattern = np.load(root+'stimuli/drosophila_optic_stimulus.npy', allow_pickle=True)
stimuli = nest.Create('poisson_generator', n=len(fibres))
for stimulus in range(len(fibres)):
	for j in fibres[stimulus]:
		nest.Connect((stimuli[stimulus],),(j+1,))
for fire in firing_pattern:
	nest.SetStatus((stimuli[int(fire[0])],), params={
		 'start': round(float(fire[1]),1),
		 'stop': round(float(fire[2]),1),
		 'rate': float(fire[3]*stim_strength)})


# Add ambient noise
ntnstatus('Creating ambient noise for circuit')
weak_noise = nest.Create('noise_generator', params={'mean':float(noise_strength), 'std':float(noise_strength*0.1)})
nest.Connect(weak_noise, list(range(1,nnum+1)), conn_spec='all_to_all')

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

# Run simulation and export spikes
ntnstatus("Running simulation of "+str(exp_length)+"ms")
nest.Simulate(float(exp_length))
#v = nest.GetStatus(voltmeter)[0]['events']['V_m']     # volts
s = nest.GetStatus(spikedetector)[0]['events']        # spikes
np.save(root+'droso_'+simulation_id+'.npy', np.array(s))
ntnsubstatus("Simulation id: "+simulation_id)
