# Based on Janelia's drosophila connectome model
# http://neuprint.janelia.org/
# https://www.biorxiv.org/content/10.1101/2020.05.18.102061v1
# Neuro-Topology Group, Institute of Mathematics, University of Aberdeen
# Authors: Jānis Lazovskis, Jason Smith
# Date: June 2020

# Packages
import numpy as np                                                                  # For current directory to read and write files
import nest                                                                 # For main simulation
import argparse                                                             # For options
from datetime import datetime
import tqdm                                                                 # For visualizing progress bar

# OWMeta imports
from owmeta_core.command import OWM
from owmeta.worm import Worm
from owmeta.neuron import Neuron
from owmeta_core.context import Context
# ****************************************************************************** #
# Read arguments
parser = argparse.ArgumentParser(
    description='C.Elegans reconstruction and validations',
    usage='python celegans.py')
parser.add_argument('--time', type=int, default=300, help='Length, in milliseconds, of experiment.')
parser.add_argument('--noise_strength', type=float, default=3, help='Strength of noise.')

parser.add_argument('--stimulus_strength', type=int, default=3000, help='Strength of stimulus.')
parser.add_argument('--stimulus_start', type=int, default=100, help='Start time of stimulus.')
parser.add_argument('--stimulus_length', type=int, default=5, help='Length of stimulus.')

parser.add_argument('--disable_build', action='store_true', help='Stops the adjacency matrix being built')
parser.add_argument('--disable_simulate', action='store_true', help='Stops the simulation being run.')
args = parser.parse_args()

# ****************************************************************************** #
# config
root = '/home/paperspace/motifs/neurotop-nest/celegans/'         # Current working directory
parallel_threads = 8                                             # Number of threads to run NEST simulation on
exc_params = {'model': 'bernoulli_synapse', 'weight': 0.3, 'delay': 1.0, 'p_transmit': 0.13} # Excitatory synapse parameters
exc_spec = {'rule': 'all_to_all'}
inh_params = {'model': 'bernoulli_synapse', 'weight': -1.5, 'delay': 0.1, 'p_transmit': 0.13} # Inhibitory synapse parameters
inh_spec = {'rule': 'all_to_all'}
gj_params = {'model': 'gap_junction', 'weight': 0.5}
gj_spec = {'rule': 'one_to_one', 'make_symmetric': True}
exp_length = args.time                                           # Length of experiment, in milliseconds
stim_strength = args.stimulus_strength
noise_strength = args.noise_strength
stim_length = args.stimulus_length                               # Length stimulus is applied for, in milliseconds
stim_start = args.stimulus_start                                 # Time at which stimulus is introduced, in milliseconds

# ****************************************************************************** #
# Auxiliary: Formatted printer for messages


def ntnstatus(message):
    print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)


def ntnsubstatus(message):
    print('    '+message, flush=True)

# ****************************************************************************** #
# Build the adjacency matrix of the graph of the C. Elegans brain


def build_matrix():
    ntnstatus('Retrieving structural information from OWMeta')
    conn = OWM().connect()
    ctx = conn(Context)(ident='http://openworm.org/data')
    net = ctx.stored(Worm).query().neuron_network()
    neurons = sorted(
        list(net.neuron()),
        key=lambda x: x.name()
    )
    neuron_names = [n.name() for n in neurons]
    n_neurons = len(neuron_names)
    neuron_ids = range(0, n_neurons)
    name2id = dict([(name, id) for name, id in zip(neuron_names, neuron_ids)])
    neuron_neurotransmitters = [n.neurotransmitter() for n in neurons]

    def is_inhibitory(synapse):
        if synapse.synclass() is not None:
            return 'GABA' in synapse.synclass()
        else:
            return 'GABA' in synapse.pre_cell().neurotransmitter()

    connections = list(net.synapse())
    exc_syn_matrix = np.zeros((n_neurons, n_neurons))
    inh_syn_matrix = np.zeros((n_neurons, n_neurons))
    gj_matrix = np.zeros((n_neurons, n_neurons))
    for connection in connections:
        pre_cell = connection.pre_cell()
        post_cell = connection.post_cell()
        weight = connection.number()
        try:
            if connection.syntype() == 'send': #Chemical synapses
                if is_inhibitory(connection):
                    inh_syn_matrix[
                        name2id[pre_cell.name()],
                        name2id[post_cell.name()]
                    ] += weight
                else:
                    exc_syn_matrix[
                        name2id[pre_cell.name()],
                        name2id[post_cell.name()]
                    ] += weight
            elif connection.syntype() == 'gapJunction':
                gj_matrix[
                    name2id[pre_cell.name()],
                    name2id[post_cell.name()]
                ] = weight
        except KeyError:
            pass # Non-neuron was found.


    # Save results in npy format
    np.save(root+"structure/celegans_chemical_excitatory.npy", exc_syn_matrix)
    np.save(root+"structure/celegans_chemical_inhibitory.npy", inh_syn_matrix)
    np.save(root+"structure/celegans_gjunctions.npy", gj_matrix)

# ****************************************************************************** #
# Build the NEST implementation of the drosophila circuit, and run a simulation
def run_nest_simulation():
    nest.ResetKernel()                                                          # Reset nest
    nest.SetKernelStatus({"local_num_threads": parallel_threads})               # Run on many threads
    # Settings for gap junctions
    nest.SetKernelStatus({'use_wfr': True,
                      'wfr_comm_interval': 1.0,
                      'wfr_tol': 0.0001,
                      'wfr_max_iterations': 15,
                      'wfr_interpolation_order': 3})
    ntnstatus('Loading celegans structural information')
    exc = np.load(root+"structure/celegans_chemical_excitatory.npy")
    inh = np.load(root+"structure/celegans_chemical_inhibitory.npy")
    gj = np.load(root+"structure/celegans_gjunctions.npy")
    nnum = exc.shape[0]

    ntnstatus('Constructing circuit')
    network = nest.Create('hh_psc_alpha_gap', n=nnum)

    def connect_syn_targets(matrix, source, conn_spec, params):
        targets = np.nonzero(matrix[source].flatten())[0]
        for target in targets:
            weight = matrix[source,target]
            target_params = params.copy()
            target_params['weight'] *= weight
            nest.Connect(
                (source+1,),
                (target+1,),
                syn_spec = params,
            )

    def connect_gap_junctions(matrix, source, conn_spec, params):
        targets = np.nonzero(matrix[source].flatten())[0]
        for target in targets:
            weight = matrix[source,target]
            target_params = params.copy()
            target_params['weight'] *= weight
            nest.Connect(
                (source+1,),
                (target+1,),
                conn_spec = conn_spec,
                syn_spec = params
            )

    for source in tqdm.tqdm(range(nnum)):
        connect_syn_targets(exc, source, exc_spec, exc_params)
        connect_syn_targets(inh, source, inh_spec, inh_params)
        connect_gap_junctions(gj, source, gj_spec, gj_params)

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
    for n in range(1, nnum+1):
        nest.Connect(voltmeter, (n,))
        nest.Connect((n,), spikedetector)
    # Create kickstart stimulus
    # kickstart = nest.Create(
    #                'dc_generator',
    #                params={
    #                        'start': float(stim_start),
    #                        'stop': float(stim_start + stim_length),
    #                        'amplitude': float(2*stim_strength)
    #                }
    #            )
    #for n in range(1, nnum+1):
        #nest.Connect(kickstart, (n,))
    # nest.Connect(kickstart, (1,))
    # Connect stimulus to circuit
    # ntnstatus('Creating thalamic nerves for stimulus')
    # fibres, firing_pattern = np.load(root+'stimuli/drosophila_'+stimulus_odor+'.npy', allow_pickle=True)
    # stimuli = nest.Create('poisson_generator', n=len(fibres))
    # for stimulus in range(len(fibres)):
    #    for j in fibres[stimulus]:
    #         nest.Connect((stimuli[stimulus],),(j+1,))
    # for fire in firing_pattern:
    #    nest.SetStatus((stimuli[int(fire[0])],), params={
    #        'start': round(float(fire[1]),1),
    #        'stop': round(float(fire[2]),1),
    #        'rate': float(fire[3]*stim_strength)})

    # Excite neuron 1
    nest.SetStatus([network[0]], {"I_e": 486.})
    # run the simulation
    simulation_id = datetime.now().strftime("%s")                               # Custom ID so files are not overwritten
    ntnstatus("Running simulation of "+str(exp_length)+"ms")
    nest.Simulate(float(exp_length))
    v = nest.GetStatus(voltmeter)[0]['events']['V_m']     # volts
    senders = nest.GetStatus(voltmeter)[0]['events']['senders']
    s = nest.GetStatus(spikedetector)[0]['events']        # spikes
    v = [v[senders == i+1] for i in range(nnum)]
    save_name = root+'simulations/celegans_volt.npy'
    spike_save_name = root + 'simulations/celegans_spikes.npy'
    np.save(save_name, np.array(v, dtype=object))
    np.save(spike_save_name, np.array(s, dtype = object))
    ntnsubstatus("Simulation name: " + save_name)


# ******************************************************************************#
# Run everything
if __name__=="__main__":
    if not args.disable_build:
        build_matrix()
    if not args.disable_simulate:
        run_nest_simulation()

