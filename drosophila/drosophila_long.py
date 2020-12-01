# Based on Janelia's drosophila connectome model
# http://neuprint.janelia.org/
# https://www.biorxiv.org/content/10.1101/2020.05.18.102061v1
# Neuro-Topology Group, Institute of Mathematics, University of Aberdeen
# Authors: Jānis Lazovskis, Jason Smith
# Date: June 2020

# Packages
from neuprint import Client, fetch_adjacencies
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
import operator


#******************************************************************************#
#config
repetitions = 500
num_stimuli = 5
time_gap = 200
root='/uoa/scratch/shared/mathematics/neurotopology/nest/neurotop-nest/drosophila/'# Current working directory
token_address = '/uoa/home/s10js8/data/janelia.key'
#root='/home/phys3smithj/Documents/software/neurotop-nest/drosophila/'          # Current working directory
#token_address = '/home/phys3smithj/Documents/data/janelia.key'                 # A text file containing the key for accessing janelia reconstruction
dataset = "hemibrain:v1.1"                                                      # Which janelia dataset to use
parallel_threads = 8                                                            # Number of threads to run NEST simulation on
inh_prop = 0.1                                                                  # Proportion of connections which are inhibitory
exc = {'weight':0.3, 'delay':1.0, 'transmit':0.13}                              # Excitatory synapse parameters
inh = {'weight':-1.5, 'delay':0.1, 'transmit':0.13}                             # Inhibitory synapse parameters
stim_strength = 50000000
noise_strength = 0
stim_length = 10                                                                # Length stimulus is applied for, in milliseconds
simulation_id = datetime.now().strftime("%s")
do_kickstart = False

def ntnstatus(message):
	print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)
def ntnsubstatus(message):
	print('    '+message, flush=True)

ntnstatus('Downloading neuron data from Janelia')
token = list(open(token_address))[0].replace('\n','')
c = Client('neuprint.janelia.org', dataset=dataset, token=token)
q = """MATCH (n :Neuron )
        WHERE n.instance <>'{}'
        RETURN n.bodyId AS bodyId, n.type AS type
        ORDER BY n.bodyId ASC
    """
results=c.fetch_custom(q)

ntnstatus('Creating stimulus')
# Get olfactory receptor neurons
ORN_types = [i for i in set(results['type']) if not i==None and i[:3]=='ORN'] # Get list of all Olfactory Receptor Neurons (ORN) types
ORNs = results[results['type'].isin(ORN_types)]                               # Get dataframe of all ORNs
type_ids = list({x[0]:set(x[1]['bodyId']) for x in list(ORNs.groupby('type'))}.items()) # Get list of pairs (ORN_type,bids), of all bids with given type
fibres_bid = [v[1] for v in type_ids]                                         # Create the fibres, each fibre is all neurons of type x
fibre_name = {type_ids[i][0]:i for i in range(len(type_ids))}                 # Maps ORN type name to the index of the fibre whose elements of the bids of said ORN type

# Map fibre values from bid to vertex id
bidToVertex = np.load(root+"structure/drosophila_bid2vertex.npy",allow_pickle=True).item()
fibres = [[bidToVertex[j] for j in i] for i in fibres_bid]

# Data downloaded from http://neuro.uni-konstanz.de/DoOR/
glomerulus = {i[0]:[j for j in i[1:] if not j==''] for i in csv.reader(open(root+'stimuli/data/receptor2glomerulus.csv'))} # Maps glomerus name to list of neuron types it contains
odor_recept = pk.load(open(root+'stimuli/data/odor2receptor.pkl','rb'))       # Maps an odor to a list where each entry is a pair (G,S), with G a glomerulus and S a strength value between -1 and 1

#Select the largest num_stimuli odors to be used
odor_size = pk.load(open(root+'stimuli/data/odor_size.pkl','rb'))
the_stimuli = [k for k, v in sorted(odor_size.items(), key=lambda item: item[1])[-num_stimuli:]]
stim_order = [i for i in range(num_stimuli) for j in range(repetitions)]
random.shuffle(stim_order)

firing_pattern = []
for k in range(len(stim_order)):
    firing_pattern.append([])
    for i in odor_recept[the_stimuli[stim_order[k]]]:
        for j in glomerulus[i[0]]:
            if float(i[1]) > 0:
                firing_pattern[-1].append((fibre_name['ORN_'+j],time_gap*(k+1),time_gap*(k+1)+stim_length,float(i[1])))

ntnstatus('Loading drosophila structural information')                      # Number of synapses in circuit
adj = np.load(root+'structure/drosophila_weighted.npy',allow_pickle=True).item().toarray() # Adjacency matrix
nnum = len(adj)                                                             # Number of neurons in circuit
snum = sum(sum(adj))

inh_syn_integer = np.random.choice(range(snum),size=int(inh_prop*snum),replace=False)   # Randomly select synapses to be inhibitory
inh_syn_binary = np.zeros(snum,dtype='int64')
for n in inh_syn_integer:
	inh_syn_binary[n] = 1

np.save(root+'/stimuli/droso_long_'+simulation_id+'.npy',np.array([fibres,firing_pattern,inh_syn_binary,stim_order,the_stimuli],dtype=object))


################################################################################
#Run Simulation

nest.ResetKernel()                                                          # Reset nest
nest.SetKernelStatus({"local_num_threads": parallel_threads,
                      "overwrite_files": True,
                      "data_path": root+'simulations/'})                    # Run on many threads

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

# Add ambient noise
ntnstatus('Creating ambient noise for circuit')
weak_noise = nest.Create('noise_generator', params={'mean':float(noise_strength), 'std':float(noise_strength*0.1)})
nest.Connect(weak_noise, list(range(1,nnum+1)), conn_spec='all_to_all')

# Connect voltage and spike readers
ntnstatus('Adding voltage and spike readers')
#voltmeter = nest.Create('voltmeter', params={
#	 'label': 'volts',
#	 'withtime': True,
#	 'withgid': True,
#	 'interval': 0.1})
spikedetector = nest.Create('spike_detector', params={
    # "to_file": True,
    # "file_extension": simulation_id+".spikes",
	'label': 'spikes',
	'withgid': True
})
for n in range(1,nnum+1):
#	nest.Connect(voltmeter,(n,))
	nest.Connect((n,),spikedetector)

#Connect stimulus to circuit
ntnstatus('Creating thalamic nerves for stimulus')
#fibres, firing_pattern, inh_syn_binary = stimulus_np.load(root+'stimuli/drosophila_long.npy', allow_pickle=True)
stimuli = nest.Create('poisson_generator', n=len(fibres))
for stimulus in range(len(fibres)):
	for j in fibres[stimulus]:
		nest.Connect((stimuli[stimulus],),(j+1,))

#Create kickstart stimulus
if do_kickstart:
    kickstart = nest.Create('poisson_generator',params={'start': 1.0, 'stop': 2.0, 'rate': float(2*stim_strength)})
    for n in range(1,nnum+1):
	    nest.Connect(kickstart,(n,))

ntnstatus("Kickstarting")
nest.Simulate(float(200))

#run the simulation
s=[[],[]]
for i in range(len(stim_order)):
    ntnstatus("Running simulation from "+str((i+1)*time_gap)+" to "+str((i+2)*time_gap)+"ms")
    for fire in firing_pattern[i]:
	    nest.SetStatus((stimuli[int(fire[0])],), params={
		 'start': round(float(fire[1]),1),
		 'stop': round(float(fire[2]),1),
		 'rate': float(fire[3]*stim_strength)})
    nest.Simulate(float(time_gap))
    spikes = nest.GetStatus(spikedetector)[0]['events']
    s[1].extend(spikes['senders'])
    s[0].extend(spikes['times'])
    nest.SetStatus(spikedetector, {'n_events': 0})

#Output is a numpy array with two columns, first column is spike time, second column is spiking gid
np.save(root+'/simulations/droso_long_'+simulation_id+'.npy', np.transpose(np.array(s)))

ntnsubstatus('Simulation name: droso_long_'+simulation_id)
ntnsubstatus('Repetitions: '+str(repetitions)+', Number of Stimuli: '+str(num_stimuli))
