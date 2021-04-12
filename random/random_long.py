# Based on Janelia's drosophila connectome model
# http://neuprint.janelia.org/
# https://www.biorxiv.org/content/10.1101/2020.05.18.102061v1
# Neuro-Topology Group, Institute of Mathematics, University of Aberdeen
# Authors: Jānis Lazovskis, Jason Smith
# Date: June 2020

# Packages
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import sys                                                                  # For current directory to read and write files
import nest                                                                 # For main simulation
import argparse                                                             # For options
from datetime import datetime                                               # For giving messages
import random
import tqdm                                                                 # For visualizing progress bar
import csv                                                                  # For loading data files
import pickle as pk
import networkx as nx

#******************************************************************************#
# Read arguments
parser = argparse.ArgumentParser(
    description='Random Nest Circuit',
    usage='python random_long.py')
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
#config
root='/uoa/home/s09js0/neurotopology/nest/neurotop-nest/random/'                # Current working directory
arallel_threads = 8                                                             # Number of threads to run NEST simulation on
inh_prop = 0.1                                                                  # Proportion of connections which are inhibitory
exc = {'weight':0.3, 'delay':1.0, 'transmit':0.13}                              # Excitatory synapse parameters
inh = {'weight':-1.5, 'delay':0.1, 'transmit':0.13}                             # Inhibitory synapse parameters

nnum = args.number_vertices
simulation_id = datetime.now().strftime("%s")                               # Custom ID so files are not overwritten

#******************************************************************************#
# Auxiliary: Formatted printer for messages
def ntnstatus(message):
    print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)
def ntnsubstatus(message):
    print('    '+message, flush=True)

#******************************************************************************#
#Build the adjacency matrix of the graph of the drosophila brain

ntnstatus('Creating Directed ER graph')
G = nx.erdos_renyi_graph(nnum,args.density,directed=True)
adj = nx.to_scipy_sparse_matrix(G,format='coo')
np.save(root+"structure/"+simulation_id+".npy",adj)
adj = adj.toarray()
exc_vert = [np.random.choice([0, 1], p=[args.inh_prop, 1-args.inh_prop]) for i in range(nnum)]


#******************************************************************************#
#Design the stimulus to  be inserted into the drosophila circuit

ntnstatus('Creating stimulus')
fibres = []
for i in range(args.number_stimuli):
    fibres.append(random.sample(range(nnum),int(nnum*args.size_stimuli)))

stim_order = [i for i in range(args.number_stimuli) for j in range(args.stimulus_reps)]
stim_start = [int(random.random()*10) for i in range(args.number_stimuli*args.stimulus_reps)]  # Start each stimulus at somepoint withing the first 10ms
# each element of firing_pattern has the form: (stim number, stim start time, stime end time, stim strength between 1 and 2)
firing_pattern = [(stim_order[i],stim_start[i],stim_start[i]+args.stimulus_length,random.uniform(1,2)) for i in range(len(stim_order))]

np.save(root+'stimuli/'+simulation_id+'.npy',np.array([fibres,firing_pattern],dtype=object))
np.save(root+'stimuli/'+simulation_id+'_order.npy',stim_order)
#******************************************************************************#
#Build the NEST implementation of the drosophila circuit, and run a simulation

nest.ResetKernel()                                                          # Reset nest
nest.SetKernelStatus({"local_num_threads": args.threads})                   # Run on many threads

snum = sum(sum(adj))                                                        # Number of synapses in circuit

inh_syn_integer = np.random.choice(range(snum),size=int(inh_prop*snum),replace=False)   # Randomly select synapses to be inhibitory
inh_syn_binary = np.zeros(snum,dtype='int64')
for n in inh_syn_integer:
	inh_syn_binary[n] = 1

ntnstatus('Constructing circuit')
network = nest.Create('izhikevich', n=nnum, params={'a':1.1})
synapse_index = 0
targets = {n:np.nonzero(adj[n])[0] for n in range(nnum)}
for source in targets.keys():
	nest.Connect((source+1,), [target+1 for target in targets[source]], conn_spec='all_to_all', syn_spec={
		'model':'bernoulli_synapse',
		'weight':exc['weight'] if exc_vert[source] else inh['weight'],
		'delay':exc['delay'] if exc_vert[source] else inh['delay'],
		'p_transmit':exc['transmit'] if exc_vert[source] else inh['transmit']})

# Add ambient noise
ntnstatus('Creating ambient noise for circuit')
weak_noise = nest.Create('noise_generator', params={'mean':float(args.noise_strength), 'std':float(args.noise_strength*0.1)})
nest.Connect(weak_noise, list(range(1,nnum+1)), conn_spec='all_to_all')

# Connect voltage and spike readers
ntnstatus('Adding voltage and spike readers')
# voltmeter = nest.Create('voltmeter', params={
# 	 'label': 'volts',
# 	 'withtime': True,
# 	 'withgid': True,
# 	 'interval': 0.1})
spikedetector = nest.Create('spike_detector', params={
	'label': 'spikes',
	'withgid': True})
for n in range(1,nnum+1):
	# nest.Connect(voltmeter,(n,))
	nest.Connect((n,),spikedetector)

#Connect stimulus to circuit
ntnstatus('Creating thalamic nerves for stimulus')
stimuli = nest.Create('poisson_generator', n=len(fibres))
for stimulus in range(len(fibres)):
	for j in fibres[stimulus]:
		nest.Connect((stimuli[stimulus],),(j+1,))

kickstart = nest.Create('poisson_generator',params={'start': 1.0, 'stop': 2.0, 'rate': float(5000000)})
for n in range(1,nnum+1):
    nest.Connect(kickstart,(n,))

ntnstatus("Kickstarting")
nest.Simulate(float(200))

ntnstatus("Running simulation of "+str(args.time)+"ms")
s=[[],[]]
for i in range(len(stim_order)):
    ntnstatus("Running simulation from "+str((i+1)*args.time)+" to "+str((i+2)*args.time)+"ms")
    fire = firing_pattern[i]
    nest.SetStatus((stimuli[int(fire[0])],), params={
	 'start': round((i+1)*args.time+float(fire[1]),1),
	 'stop': round((i+1)*args.time+float(fire[2]),1),
	 'rate': float(fire[3]*args.stimulus_strength)})
    nest.Simulate(float(args.time))

#Output is a numpy array with two columns, first column is spike time, second column is spiking gid
spikes = nest.GetStatus(spikedetector)[0]['events']
s[1].extend(spikes['senders'])
s[0].extend(spikes['times'])
np.save(root+'/simulations/'+simulation_id+'.npy', np.transpose(np.array(s)))

ntnsubstatus('Simulation name: '+simulation_id)
ntnsubstatus('Arguments used:')
print(args)
