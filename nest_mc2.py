# Biological neural network model in NEST simulator. Based on BlueBrain neocortical microcircuit and experiments from Frontiers paper.
# Neuro-Topology Group, Institute of Mathematics, University of Aberdeen
# Authors: Jānis Lazovskis, Jason Smith
# Date: March 2020

# Import packages
import pylab
import itertools
import matplotlib as mpl                                                    # For plotting
import matplotlib.pyplot as plt                                             # For plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable                     # For plotting, to add colorbar nicely
import numpy as np                                                          # For reading of files
import sys
import nest                                                                 # For main simulation
import argparse                                                             # For options
from datetime import datetime                                               # For giving messages

# Auximilary functions
def ntnstatus(message):
	print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)


# Read arguments
parser = argparse.ArgumentParser(
    description='Simplified Blue Brain Project reconstructions and validations',
    usage='python nest_mc2.py [--fibres=fibres.npy] [--stimulus=stim.npy] [--time=100] [--title="Plot title"]')
parser.add_argument('--fibres', type=str, default='fibres.npy', help='Filename of fibres in folder "stimuli". List of lists of GIDs. Length is number of thalamic fibers.')
parser.add_argument('--stimulus', type=str, default='constant_firing.npy', help='Filename of firing pattern of stimulus in folder "stimuli". List of tuples (index,start,stop,rate). Index has to be within range of length of fibres option.')
parser.add_argument('--time', type=int, default=100, help='Length, in milliseconds, of experiment. Must be an integer. Default is 100.')
parser.add_argument('--title', type=str, default='Spikemeter and voltmeter reports', help='Title for plot produced at the end.')
args = parser.parse_args()

# Set up
#nest.set_verbosity("M_ERROR")                                              # Uncomment this to make NEST quiet
nest.ResetKernel()                                                          # Reset nest
root = sys.argv[0][:-11]                                                    # Current working directory
                                        
# Load circuit info
mc2_address = root+'structure/adjmat_mc2.npz'                               # Address of mc2 adjacency matrix
adj = np.load(mc2_address)
mc2_edges = list(zip(adj['row'], adj['col']))                               # Get edges between neurons
nnum = 31346                                                                # Number of neurons in circuit

# Load stimulus info
fibres_address = root+'stimuli/'+args.fibres                                # Address of thalamic nerve fibres data
fibres = np.load(fibres_address,allow_pickle=True)                          # Get which neurons the fibres connect to
stimulus_address = root+'stimuli/'+args.stimulus
firing_pattern = np.load(stimulus_address,allow_pickle=True)                # When the nerve fibres fire, each element of the form (fibre_id,start_time,end_time,firing_rate)
stim_strength = 1000000

# Declare parameters
augold2 = (1., 0.866667, 0.)                                                # Declare color scheme
weight = 20.0                                                               # Maximumax weight of synapses
delay = 1.0                                                                 # Delay between synapses
exp_length = args.time                                                      # length of experiment, in milliseconds

# Create the circuit
ntnstatus('Constructing circuit')
network = nest.Create('iaf_cond_exp_sfa_rr', n=nnum, params=None)
for i in range(len(mc2_edges[0])):
    nest.Connect((mc2_edges[0][i]+1,),(mc2_edges[1][i]+1,),
                    syn_spec={'weight': np.random.random()*weight, 'delay': delay})

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
    nest.SetStatus((stimuli[fire[0]],), params={
         'start':float(fire[1]),
         'stop':float(fire[2]),
         'rate': float(fire[3])*stim_strength})

# Run simulation
ntnstatus("Running simulation of "+str(exp_length)+"ms")
nest.Simulate(float(exp_length))

# Print reports of experiment
ntnstatus("Creating spike and volt plots")
stims = nest.GetStatus(spikedetector)[0]['events']
volts = nest.GetStatus(voltmeter)[0]['events']['V_m']

fig = plt.figure(figsize=(15,5)) # default is (8,6)
fig.suptitle(args.title,fontsize=18)

ax_stim = fig.add_subplot(2,1,1)
ax_stim.scatter(stims['times'], stims['senders'], s=1, marker="+")
ax_stim.set_ylabel('neuron index')
ax_stim.set_xticks([])
ax_stim.set_ylim(1,nnum)
plt.gca().invert_yaxis()
ax_stim.set_xlim(1,exp_length)

ax_volt = fig.add_subplot(212)
v = ax_volt.imshow(np.transpose(np.array(volts).reshape(int(exp_length-1),nnum)), cmap=plt.cm.Spectral_r, interpolation='None', aspect="auto")
ax_volt.set_ylabel('neuron index')
ax_volt.set_xlabel('time in ms')

fig.colorbar(v, ax=[ax_stim,ax_volt], orientation='vertical', label="voltage")
plt.savefig(root+'report.png')
