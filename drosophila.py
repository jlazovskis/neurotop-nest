# Based on BlueBrain neocortical microcircuit
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

# Auxiliary: Formatted printer for messages
def ntnstatus(message):
	print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)

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
adj = load_npz(root+'structure/weighted_adjmat_drosophila.npz').toarray().astype('int64')   # Adjacency matrix
#exc = 

# Declare parameters
syn_weight = 1.0                                                            # Weight of excitatory synapses
inh_weight = -1.0                                                           # Weight of inhibitory synapses
delay = 0.1                                                                 # Delay between neurons (used as a multiplying factor)
exp_length = args.time                                                      # Length of experiment, in milliseconds

# Create the circuit
ntnstatus('Constructing circuit')
network = nest.Create('izhikevich', n=nnum, params={'a':1.1})
targets = {n:np.nonzero(adj[n])[0] for n in range(nnum)}
for source in targets.keys():
	nest.Connect((source+1,), [target+1 for target in targets[source]], conn_spec='all_to_all', syn_spec={
		'model': 'bernoulli_synapse',
		'weight': syn_weight,# if exc[source] else inh_weight
		'delay': delay,
		'p_transmit': 0.1})

# Define stimulus and connect it to neurons
#ntnstatus('Creating spikes for stimulus')
#stimulus = nest.Create('spike_generator', params={'spike_times': [50.0,51.0,52.0,500.0], 'precise_times':True})
#thalamic_targets = np.random.choice(list(range(nnum)), size=nnum//2)
#nest.Connect(stimulus, [target+1 for target in thalamic_targets], conn_spec='all_to_all')

# Define stimulus and connect it to neurons
ntnstatus('Creating thalamic nerves for stimulus')
#stimulus_times = [np.random.rand()*exp_length for i in range(10)]
#stimulus_times.sort()
#stimulus_times = [i*10.0 for i in range(1,10)]
thalamic_targets = np.random.choice(list(range(nnum)), size=nnum//5)
stimulus1 = nest.Create('poisson_generator', n=len(thalamic_targets))
for thalamus in range(len(thalamic_targets)):
	nest.Connect((stimulus1[thalamus],),(thalamic_targets[thalamus]+1,))
for fire in range(len(thalamic_targets)):
	nest.SetStatus((stimulus1[fire],), params={
		 'start': 30.0,
		 'stop': 32.0,
		 'rate': 15000.0})

thalamic_targets = np.random.choice(list(range(nnum)), size=nnum//5)
stimulus2 = nest.Create('poisson_generator', n=len(thalamic_targets))
for thalamus in range(len(thalamic_targets)):
	nest.Connect((stimulus2[thalamus],),(thalamic_targets[thalamus]+1,))
for fire in range(len(thalamic_targets)):
	nest.SetStatus((stimulus2[fire],), params={
		 'start': 60.0,
		 'stop': 62.0,
		 'rate': 15000.0})


# Add ambient noise
ntnstatus('Creating ambient noise for circuit')
weak_noise = nest.Create('noise_generator', params={'mean':2.0, 'std':0.5})
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