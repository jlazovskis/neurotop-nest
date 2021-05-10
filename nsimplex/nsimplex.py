# A model using n-dimensional simplices
# Date: April 2021

# Packages
import numpy as np                                               # For current directory to read and write files
import nest                                                      # For main simulation
import argparse                                                  # For options
from datetime import datetime
from pathlib import Path
# ******************************************************************************#
# ******************************************************************************#
# Formatted printer for messages


def ntnstatus(message):
    print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)


def ntnsubstatus(message):
    print('    '+message, flush=True)


# ******************************************************************************#
def simulate(args):
    # Configuration
    ntnstatus('Configuring')
    root = args.root                                                               # Current working directory
    parallel_threads = 8                                                           # Number of threads to run NEST simulation on
    exc = {'weight':10.0, 'delay':0.5, 'transmit':0.13}                             # Excitatory synapse parameters
    inh = {'weight':-1.5, 'delay':0.1, 'transmit':0.13}                            # Inhibitory synapse parameters

    simulation_id = args.id_prefix + datetime.now().strftime("%s")                                  # Custom ID so files are not overwritten


    #******************************************************************************#
    # Load circuit and stimulus
    ntnstatus('Loading circuit and stimulus from ' + root+'/structure/'+args.exc_adj+'.npy')
    adj = np.load(root+'/structure/'+args.exc_adj+'.npy', allow_pickle = True)                          # Adjacency matrix of circuit
    if args.inh_adj:
        adj_inh = np.load(root+'/structure/'+args.inh_adj+'.npy', allow_pickle = True)                  # Inhibitory synapse mask
    else:
        adj_inh = np.zeros_like(adj)
    nnum = len(adj)
    stimulus_targets = {
        'all':[list(range(nnum))],
        'source':[[0]],
        'sink':[[nnum-1]]
    }

    try:
        fibres = stimulus_targets[args.stimulus_targets]
    except KeyError:
        ntnsubstatus("Stimulus targets config " + args.stimulus_targets+ " not found. Connecting stimulus to all")
        fibres = stimulus_targets['all']

    firing_pattern = [(
                    0,
                    args.stimulus_start,
                    args.stimulus_start + args.stimulus_length,
                    float(args.stimulus_strength)
        )]    # (fibre index, start time, stop time, rate)

    #******************************************************************************#
    # Build the circuit
    nest.ResetKernel()                                                             # Reset nest
    nest.SetKernelStatus({"local_num_threads": args.threads})                      # Run on many threads

    snum = sum(sum(adj))                                                           # Number of synapses in circuit
    for source, target in []:
        adj_inh[source][target] = True

    ntnstatus('Constructing circuit')
    network = nest.Create('izhikevich', n=nnum, params={'a':1.1})
    for source, target in zip(*np.nonzero(adj)):
        inh_syn = adj_inh[source][target]
        nest.Connect((source+1,), (target+1,), conn_spec='all_to_all', syn_spec={
            'model':'static_synapse',
            'weight':exc['weight'] if not inh_syn else inh['weight'],
            'delay':exc['delay'] if not inh_syn else inh['delay']})


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
    ntnstatus('Creating stimulus for stimulus')
    fire = firing_pattern[0]
    if args.stimulus_type == 'poisson':
        stimuli = nest.Create('poisson_generator', n=len(fibres))
        nest.SetStatus((stimuli[int(fire[0])],), params={
             'start': round(float(fire[1]),1),
             'stop': round(float(fire[2]),1),
             'rate': float(fire[3])})
    if args.stimulus_type == 'dc':
        stimuli = nest.Create('dc_generator', n=len(fibres))
        nest.SetStatus((stimuli[int(fire[0])],), params={
             'start': round(float(fire[1]),1),
             'stop': round(float(fire[2]),1),
             'amplitude': float(fire[3])})
    if args.stimulus_type == 'ac':
        stimuli = nest.Create('ac_generator', n=len(fibres))
        nest.SetStatus((stimuli[int(fire[0])],), params={
             'start': round(float(fire[1]),1),
             'stop': round(float(fire[2]),1),
             'amplitude': float(fire[3]),
             'frequency': args.stimulus_frequency})

    for fibre_index in range(len(fibres)):
        fibre = fibres[fibre_index]
        for target in fibre:
            nest.Connect((stimuli[fibre_index],),(target+1,))


    #******************************************************************************#
    # Run a simulation
    ntnstatus("Running simulation of "+str(args.time)+"ms")
    nest.Simulate(float(args.time))
    spikes = nest.GetStatus(spikedetector)[0]['events']
    volts = nest.GetStatus(voltmeter)[0]['events']

    # Save results
    volt_save_path = root+'/simulations/'+str(args.save_name)+'_'+simulation_id+'-volts.npy'
    spikes_save_path = root+'/simulations/'+str(args.save_name)+'_'+simulation_id+'-spikes.npy'
    Path(spikes_save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(
        spikes_save_path,
        np.array([spikes['senders'],spikes['times']])
        )
    np.save(
        volt_save_path,
        np.array([volts['senders'],volts['times'],volts['V_m']])
        )

    ntnsubstatus('Simulation name: '+args.save_name + simulation_id)
    ntnsubstatus('Arguments used:')
    print(args)
    return simulation_id

if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser(
    description='n-simplex Nest Circuit',
    usage='python nsimplex.py')

    parser.add_argument('--root', type=str, default='.', help='Root directory for importing and exporting files')
    parser.add_argument('--exc_adj', type=str, default='3simplex/3simplex', help='Path to the circuit excitatory syn matrix, without .npy.')
    parser.add_argument('--inh_adj', type=str, default="", help='Path to the circuit inhibitory syn matrix mask, without .npy.')
    parser.add_argument('--save_name', type=str, default='3simplex', help='Path to save the results')
    parser.add_argument('--stimulus_targets', type=str, default="all", help='Stimulus targets. \'sink\', \'source\', \'all\' are supported')
    parser.add_argument('--stimulus_type', type=str, default="poisson", help='Stimulus type. \'dc\', \'ac\', \'poisson\', \'poisson_parrot\' are supported.')
    parser.add_argument('--stimulus_frequency', type=float, default=1., help='Stimulus frequency in ac case. Unusued for other stimuli.')
    parser.add_argument('--noise_strength', type=float, default=3., help='Strength of noise.')
    parser.add_argument('--stimulus_strength', type=int, default=40, help='Strength of stimulus.')
    parser.add_argument('--stimulus_length', type=int, default=100, help='Length of stimulus.')
    parser.add_argument('--stimulus_start', type=int, default=5, help='Length of stimulus.')
    parser.add_argument('--time', type=int, default=200, help='Length, in milliseconds, between stimuli. Must be an integer. Default is 200.')
    parser.add_argument('--threads', type=int, default=40, help='Number of parallel thread to use. Must be an integer. Default is 40.')
    parser.add_argument('--id_prefix', type=str, default='', help='Prefix for sim ID')
    keyword_args = parser.parse_args()

    simulate(keyword_args)
