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

#******************************************************************************#
# Read arguments
parser = argparse.ArgumentParser(
	description='Drosophila reconstruction and validations',
	usage='python drosophila.py')
parser.add_argument('--time', type=int, default=300, help='Length, in milliseconds, of experiment.')
parser.add_argument('--noise_strength', type=float, default=3, help='Strength of noise.')

parser.add_argument('--stimulus', type=str, default='2-3-butanedione', help='Stimulus to use. Default is 2-3-butanedione which is the strongest.') # Avaible odors can be obtained by: pickle.load(open(root+'stimuli/data/odor2receptor.pkl','rb')).keys()
parser.add_argument('--stimulus_strength', type=int, default=50000, help='Strength of stimulus.')
parser.add_argument('--stimulus_start', type=int, default=100, help='Start time of stimulus.')
parser.add_argument('--stimulus_length', type=int, default=5, help='Length of stimulus.')

parser.add_argument('--disable_build', action='store_true', help='Stops the adjacency matrix being built and loads "structure/drosophila_weighted.npy".')
parser.add_argument('--disable_design', action='store_true', help='Stops the stimulus being designed and loads "stimuli/drosophila_olfact.npy".')
parser.add_argument('--disable_simulate', action='store_true', help='Stops the simulation being run.')
args = parser.parse_args()

#******************************************************************************#
#config
root='/uoa/scratch/shared/mathematics/neurotopology/nest/neurotop-nest/drosophila/'# Current working directory
token_address = '/uoa/home/s10js8/data/janelia.key'                             # A text file containing the key for accessing janelia reconstruction, please 
dataset = "hemibrain:v1.1"                                                      # Which janelia dataset to use
parallel_threads = 8                                                            # Number of threads to run NEST simulation on
inh_prop = 0.1                                                                  # Proportion of connections which are inhibitory
exc = {'weight':0.3, 'delay':1.0, 'transmit':0.13}                              # Excitatory synapse parameters
inh = {'weight':-1.5, 'delay':0.1, 'transmit':0.13}                             # Inhibitory synapse parameters
exp_length = args.time                                                          # Length of experiment, in milliseconds
stim_strength = args.stimulus_strength
noise_strength = args.noise_strength
stim_length = args.stimulus_length                                              # Length stimulus is applied for, in milliseconds
stim_start = args.stimulus_start                                                # Time at which stimulus is introduced, in milliseconds
stimulus_odor = args.stimulus

#******************************************************************************#
# Auxiliary: Formatted printer for messages
def ntnstatus(message):
	print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)
def ntnsubstatus(message):
	print('    '+message, flush=True)

#******************************************************************************#
#Build the adjacency matrix of the graph of the drosophila brain
def build_matrix(results):
    ntnstatus('Downloading structural information from Janelia')
    bids = list(results['bodyId'].values)                                                   # Get bids of neurons to consider
    neuron_df, conn_df = fetch_adjacencies(bids,bids)                                       # Get all synapses between neurons, and their weight
    conn_df2=conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum() # Combine repeated edges and add their weights

    #Dictionary of bid to vertex id
    bodyIdtoId={bids[i]:i for i in range(len(bids))}

    #Save results in coo format
    row=[bodyIdtoId[i] for i in conn_df2['bodyId_pre'].values]
    col=[bodyIdtoId[i] for i in conn_df2['bodyId_post'].values]
    data=[i for i in conn_df2['weight'].values]
    N=len(bids)
    mcoo = sparse.coo_matrix((data, (row, col)), shape=(N, N))
    np.save(root+"structure/drosophila_weighted.npy",mcoo)
    np.save(root+"structure/drosophila_bid2vertex.npy",bodyIdtoId)


#******************************************************************************#
#Design the stimulus to  be inserted into the drosophila circuit
def design_stimuli(results):
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
    odor_recept = pk.load(open(root+'stimuli/data/odor2receptor.pkl','rb'))   # Maps an odor to a list where each entry is a pair (G,S), with G a glomerulus and S a strength value between -1 and 1

    # Each stimulus consists of the tuple (fibre,start_time,end_time,strength)
    stimulus = []
    for i in odor_recept[stimulus_odor]:
        for j in glomerulus[i[0]]:
            if float(i[1]) > 0:                                                 # Some of the strengths are negative, not sure what that means, so for now ignore
                stimulus.append((fibre_name['ORN_'+j],stim_start,stim_start+stim_length,float(i[1])))

    np.save(root+'/stimuli/drosophila_'+stimulus_odor+'.npy',np.array([fibres,stimulus],dtype=object))

#******************************************************************************#
#Build the NEST implementation of the drosophila circuit, and run a simulation
def run_nest_simulation():
    nest.ResetKernel()                                                          # Reset nest
    nest.SetKernelStatus({"local_num_threads": parallel_threads})               # Run on many threads

    ntnstatus('Loading drosophila structural information')                      # Number of synapses in circuit
    adj = np.load(root+'structure/drosophila_weighted.npy',allow_pickle=True).item().toarray() # Adjacency matrix
    nnum = len(adj)                                                             # Number of neurons in circuit
    snum = sum(sum(adj))                                                        # Number of synapses in circuit

    inh_syn_integer = np.random.choice(range(snum),size=int(inh_prop*snum),replace=False)   # Randomly select synapses to be inhibitory
    inh_syn_binary = np.zeros(snum,dtype='int64')
    for n in inh_syn_integer:
    	inh_syn_binary[n] = 1

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

    #Create kickstart stimulus
    # kickstart = nest.Create('poisson_generator',params={'start': 1.0, 'stop': 2.0, 'rate': float(2*stim_strength)})
    # for n in range(1,nnum+1):
    # 	nest.Connect(kickstart,(n,))

    #Connect stimulus to circuit
    ntnstatus('Creating thalamic nerves for stimulus')
    fibres, firing_pattern = np.load(root+'stimuli/drosophila_'+stimulus_odor+'.npy', allow_pickle=True)
    stimuli = nest.Create('poisson_generator', n=len(fibres))
    for stimulus in range(len(fibres)):
    	for j in fibres[stimulus]:
    		nest.Connect((stimuli[stimulus],),(j+1,))
    for fire in firing_pattern:
    	nest.SetStatus((stimuli[int(fire[0])],), params={
    		 'start': round(float(fire[1]),1),
    		 'stop': round(float(fire[2]),1),
    		 'rate': float(fire[3]*stim_strength)})

    #run the simulation
    simulation_id = datetime.now().strftime("%s")                               # Custom ID so files are not overwritten
    ntnstatus("Running simulation of "+str(exp_length)+"ms")
    nest.Simulate(float(exp_length))
    #v = nest.GetStatus(voltmeter)[0]['events']['V_m']     # volts
    s = nest.GetStatus(spikedetector)[0]['events']        # spikes
    np.save(root+'/simulations/droso_'+stimulus_odor+'_'+simulation_id+'.npy', np.array(s,dtype=object))
    ntnsubstatus("Simulation name: droso_"+stimulus_odor+"_"+simulation_id)



#******************************************************************************#
#Run everything
if __name__=="__main__":
    # If we need to download data from Janelia do so
    if not args.disable_build or not args.disable_design:
        ntnstatus('Downloading neuron data from Janelia')
        token = list(open(token_address))[0].replace('\n','')
        c = Client('neuprint.janelia.org', dataset=dataset, token=token)
        q = """MATCH (n :Neuron )
                WHERE n.instance <>'{}'
                RETURN n.bodyId AS bodyId, n.type AS type
                ORDER BY n.bodyId ASC
            """
        results=c.fetch_custom(q)
    if not args.disable_build:
        build_matrix(results)
    if not args.disable_design:
        design_stimuli(results)
    if not args.disable_simulate:
        run_nest_simulation()

