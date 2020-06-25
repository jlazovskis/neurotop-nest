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
adj = load_npz(root+'structure/weighted_adjmat_drosophila.npz').toarray().astype('int64')   # Adjacency matrix
inh = np.load(root+'structure/drosophila_inhibitory.npy')                                   # Binarty array indicating whether or not neuron is inhibitory

# Declare parameters
exc_weight = 0.25                                                           # Weight (as a factor) of excitatory synapses
inh_weight = -5.0                                                           # Weight (as a factor) of inhibitory synapses
unique_weights = 30                                                         # Number of different synapse weights to observe. Starts with smallest nonzero weight. Max is 108. 
delay = 0.1                                                                 # Delay between neurons (used as a multiplying factor)
exp_length = args.time                                                      # Length of experiment, in milliseconds
stim_strength = 500
noise_strength = 3.75

# Create the circuit
ntnstatus('Constructing circuit')
network = nest.Create('izhikevich', n=nnum, params={'a':1.1})
weights = {n:np.unique(adj[n],return_counts=True) for n in range(nnum)}
#targets = {n:np.nonzero(adj[n])[0] for n in range(nnum)}
targets = {n:{w:np.where(adj[n] == w)[0] for w in weights[n][0][1:unique_weights+1]} for n in range(nnum)}
largest_weights = []
for source in targets.keys():
	if len(weights[source][0]) > unique_weights+1:
		weight_ceiling = weights[source][0][unique_weights]
		targets[source][weight_ceiling] = np.where(adj[source] >= weight_ceiling)[0]
		largest_weights.append(weight_ceiling)
	for weight in targets[source].keys():
		nest.Connect((source+1,), [target+1 for target in targets[source][weight]], conn_spec='all_to_all', syn_spec={
			'model': 'bernoulli_synapse',
			'weight': weight*inh_weight if inh[source] else weight*exc_weight,
			'delay': delay,
			'p_transmit': 0.1})
if largest_weights != []:
	missed = sum([sum(weights[n][1][unique_weights+1:]) for n in range(nnum)])
	ntnsubstatus('Clipped weights of '+str(missed)+' synapses ({0:.3f} percent)'.format(100*missed/3413160))
	ntnsubstatus('Average clipped weight was {0:.2f}'.format(sum(largest_weights)/len(largest_weights)))
else:
	ntnsubstatus('All unique synapse weights observed')

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
# thalamic_targets = np.random.choice(list(range(nnum)), size=nnum//5)
# stimulus1 = nest.Create('poisson_generator', n=len(thalamic_targets))
# for thalamus in range(len(thalamic_targets)):
# 	nest.Connect((stimulus1[thalamus],),(thalamic_targets[thalamus]+1,))
# for fire in range(len(thalamic_targets)):
# 	nest.SetStatus((stimulus1[fire],), params={
# 		 'start': 30.0,
# 		 'stop': 32.0,
# 		 'rate': 15000.0})
#
# thalamic_targets = np.random.choice(list(range(nnum)), size=nnum//5)
# stimulus2 = nest.Create('poisson_generator', n=len(thalamic_targets))
# for thalamus in range(len(thalamic_targets)):
# 	nest.Connect((stimulus2[thalamus],),(thalamic_targets[thalamus]+1,))
# for fire in range(len(thalamic_targets)):
# 	nest.SetStatus((stimulus2[fire],), params={
# 		 'start': 60.0,
# 		 'stop': 62.0,
# 		 'rate': 15000.0})
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
