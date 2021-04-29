# A model using n-dimensional simplices
# Date: April 2021

# Packages
import numpy as np
import sys                                                                  # For current directory to read and write files
import nest                                                                 # For main simulation
import argparse                                                             # For options
from datetime import datetime                                               # For giving messages
import random

#******************************************************************************#
# Read arguments
parser = argparse.ArgumentParser(
    description='n-simplex Nest Circuit',
    usage='python nsimplex.py')
parser.add_argument('--root', type=str, default='.', help='Root directory for importing and exporting files')
parser.add_argument('--circuit', type=str, default='3simplex', help='Name of circuit file, without .npy.')

parser.add_argument('--noise_strength', type=float, default=3, help='Strength of noise.')
parser.add_argument('--number_vertices', type=int, default=1000, help='Number of vertices in the graph.')
parser.add_argument('--density', type=float, default=0.08, help='Density of ER graph. Takes a value between 0 and 1.')
parser.add_argument('--inh_prop', type=float, default=0.1, help='Proportion of inhibitory vertices. Takes a value between 0 and 1.')

parser.add_argument('--number_stimuli', type=int, default=8, help='Number of stimuli.')
parser.add_argument('--size_stimuli', type=float, default=0.1, help='Number of vertices stimulated by each stimulus as a proportion of all vertices. Takes a value between 0 and 1.')
parser.add_argument('--stimulus_strength', type=int, default=50000, help='Strength of stimulus.')
parser.add_argument('--stimulus_length', type=int, default=5, help='Length of stimulus.')
parser.add_argument('--stimulus_reps', type=int, default=500, help='Number of times each stimulus is repeated.')
parser.add_argument('--time', type=int, default=200, help='Length, in milliseconds, between stimuli. Must be an integer. Default is 200.')
parser.add_argument('--threads', type=int, default=40, help='Number of parallel thread to use. Must be an integer. Default is 40.')

args = parser.parse_args()

#******************************************************************************#
# Formatted printer for messages
def ntnstatus(message):
    print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)
def ntnsubstatus(message):
    print('    '+message, flush=True)


#******************************************************************************#
# Configuration
ntnstatus('Configuring')
root = args.root                                                               # Current working directory
parallel_threads = 8                                                           # Number of threads to run NEST simulation on
exc = {'weight':0.3, 'delay':1.0, 'transmit':0.13}                             # Excitatory synapse parameters
inh = {'weight':-1.5, 'delay':0.1, 'transmit':0.13}                            # Inhibitory synapse parameters

simulation_id = datetime.now().strftime("%s")                                  # Custom ID so files are not overwritten


#******************************************************************************#
# Load circuit and stimulus
ntnstatus('Loading circuit and stimulus')
adj = np.load(root+'/structure/'+args.circuit+'.npy')                          # Adjacency matrix of circuit
nnum = len(adj)
fibres = [list(range(nnum))]
firing_pattern = [(0,5,10, float(5000000))]                                    # (fibre index, start time, stop time, rate)

#np.save(root+'stimuli/'+simulation_id+'.npy',np.array([fibres,firing_pattern],dtype=object))
#np.save(root+'stimuli/'+simulation_id+'_order.npy',stim_order)


#******************************************************************************#
# Build the circuit
nest.ResetKernel()                                                             # Reset nest
nest.SetKernelStatus({"local_num_threads": args.threads})                      # Run on many threads

snum = sum(sum(adj))                                                           # Number of synapses in circuit
adj_inh = np.zeros_like(adj)                                                   # Inhibitory synapse mask
for source, target in []:
	adj_inh[source][target] = True

ntnstatus('Constructing circuit')
network = nest.Create('izhikevich', n=nnum, params={'a':1.1})
for source, target in zip(*np.nonzero(adj)):
	inh_syn = adj_inh[source][target]
	nest.Connect((source+1,), (target+1,), conn_spec='all_to_all', syn_spec={
		'model':'bernoulli_synapse',
		'weight':exc['weight'] if inh_syn else inh['weight'],
		'delay':exc['delay'] if inh_syn else inh['delay'],
		'p_transmit':exc['transmit'] if inh_syn else inh['transmit']})


#******************************************************************************#
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


#******************************************************************************#
# Connect stimulus to circuit
ntnstatus('Creating thalamic nerves for stimulus')
stimuli = nest.Create('poisson_generator', n=len(fibres))
for fibre_index in range(len(fibres)):
	fibre = fibres[fibre_index]
	for target in fibre:
		nest.Connect((stimuli[fibre_index],),(target+1,))
for fire in firing_pattern:
	nest.SetStatus((stimuli[int(fire[0])],), params={
		 'start': round(float(fire[1]),1),
		 'stop': round(float(fire[2]),1),
		 'rate': float(fire[3])*args.stimulus_strength})


#******************************************************************************#
# Run a simulation
ntnstatus("Running simulation of "+str(args.time)+"ms")
nest.Simulate(float(args.time))
spikes = nest.GetStatus(spikedetector)[0]['events']
volts = nest.GetStatus(voltmeter)[0]['events']

# Save results
np.save(
	root+'/simulations/'+str(args.circuit)+'_'+simulation_id+'-spikes.npy',
	np.array([spikes['senders'],spikes['times']])
	)
np.save(
	root+'/simulations/'+str(args.circuit)+'_'+simulation_id+'-volts.npy',
	np.array([volts['senders'],volts['times'],volts['V_m']])
	)

ntnsubstatus('Simulation name: '+simulation_id)
ntnsubstatus('Arguments used:')
print(args)