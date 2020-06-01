# Packages
import h5py                                                                 # For exporting h5 files
import numpy as np                                                          # For reading of files
import pandas as pd                                                         # For reading of files
from functools import reduce                                                # For combining simplices
import matplotlib as mpl                                                    # For plotting
import matplotlib.pyplot as plt                                             # For plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable                     # For plotting, to add colorbar nicely
import scipy.sparse                                                         # For exporting transimssion response matrices
import subprocess                                                           # For counting simplices and running flagser

nnum = 31346                                                                # Number of neurons in circuit

##
## Functions that work with standalone inputs
##

# Output spike only plot
def make_spike_plot(name,length):
	s = np.load(name+'.npy',allow_pickle=True)[True][0]
	fig = plt.figure(figsize=(20,6))
	plt.gca().invert_yaxis()
	plt.scatter(s['times'], s['senders'], s=1, marker="s", c=[(0.8353, 0.0, 0.1961) for i in range(len(s['times']))], edgecolors='none', alpha=.8)
#	plt.set_ylabel('neuron index')
#	ax_spikes.set_ylim(1,simulation.neurons)
#	ax_spikes.set_xlim(0,simulation.length)
	plt.xlim(0,length)
	plt.tight_layout()
	plt.savefig(name+'.png', dpi=200)


# Transimssion response matrices from spike file
def make_tr_fromspikes(name, length, t1, t2):
	s = np.load(name+'.npy',allow_pickle=True)[True][0]
	adj = scipy.sparse.load_npz('structure/adjmat_mc2.npz').toarray()
	active_edges = []
	for step in range(int(length//t1)):
		print('Looking at time step '+str(step),flush=True)
		cur_edges = []
		for spike_index in np.where(abs(s['times']-(step+.5)*t1) < t1/2)[0]:
			spike_time = s['times'][spike_index]
			spiker = s['senders'][spike_index]-1
			for alsospiked in np.intersect1d(np.nonzero(adj[spiker])[0], np.array([n-1 for n in s['senders'][np.where(abs(s['times']-spike_time-t2/2) < t2/2)[0]]])):
				cur_edges.append(tuple({spiker,alsospiked}))
		active_edges.append(cur_edges)
		print('Active edges: '+str(len(cur_edges)), flush=True)
	print('Compressing edge lists into npz file', flush=True)
	np.savez_compressed(name+'.npz', *active_edges)


# Run flagser on transmission response edge lists
def flag_tr(name, flagser='/home/jlv/flagser/flagser'):
	tr = load name
	bettis = []
	for step in range(len(tr)):
		cur_bettis = []
		# Make file for Flagser
		print('Reading step '+str(step),flush=True)
		f = open('step'+str(step)+'.in','w'); f.write('dim 0\n')
		for i in range(nnum):
			f.wwrite('0 ')
		f.write('\ndim 1\n')
		for edge in tr[step]:
			f.write(str(edge[0]+1)+' '+str(edge[1]+1)+'\n')
		f.close()
		# Run flagser and read output file
		print('Flagging step '+str(step),flush=True)
		cmd = subprocess.Popen([flagser, '--out', 'step'+str(step)+'.out', 'step'+str(step)+'.in'], stdout=subprocess.DEVNULL); cmd.wait()
		g = open('step'+str(step)+'.out','r'); L = g.readlines(); g.close()
		bettis.append(np.array(list(map(lambda x: int(x), L[1][:-1].split(' ')[1:]))))
		# Remove files
		subprocess.Popen(['rm', 'step'+str(step)+'.in'], stdout=subprocess.DEVNULL)
	np.save(name+'_TR.npy',np.array(bettis))

def make_betti_curves(name):
	fig = plt.figure(figsize=(8,6)) # default is (8,6)
	mpl.rcParams['axes.spines.right'] = False; mpl.rcParams['axes.spines.top'] = False
	ax1 = fig.add_subplot(4,4,1)

	plt.savefig(name+'_TR.png',dpi=200)


# Make visual plot of locations of neurons spiking
def make_loc_plot(volt_file,number_of_steps):
	# Load data
	data = np.transpose(np.load(volt_file).reshape(int(number_of_steps-1),31346))
	locs = pd.read_pickle('/home/jlv/Documents/Darbi - algoti/2019-08 University of Aberdeen/Projects/2020-03-10 New distances/2020-05-02-active-neurons/mc2_locs.pkl')
	ranges = {'x':[min(locs.iloc[0]),max(locs.iloc[0])], 'y':[min(locs.iloc[1]),max(locs.iloc[1])], 'z':[min(locs.iloc[2]), max(locs.iloc[2])]}

	# Set up figure
	fig = plt.figure(figsize=(20,10)) # default is (8,6)
	siz = 10
	transp = .01
	for s in ['axes.spines.left','axes.spines.right','axes.spines.top','axes.spines.bottom']:
		mpl.rcParams[s] = False
	yz = fig.add_subplot(1,3,1); xz = fig.add_subplot(1,3,2); xy = fig.add_subplot(1,3,3)

	# Plot figure
#	for t in range(number_of_steps):
	t = 600
	voltrange_raw = max([data[n][t] for n in range(31346)]) - min([data[n][t] for n in range(31346)])
	voltrange = 0 if voltrange_raw < 0.0001 else voltrange_raw
	for projh,projv,ax in zip(['y','x','x'],['z','z','y'],[yz,xz,xy]):
		print('Plotting plane '+projh+projv,flush=True)
		horizontal = []; vertical = []; shade = [];
		for n in range(31346):
			horizontal.append(locs[n][projh]); vertical.append(locs[n][projv]); shade.append(data[n][t]/voltrange if voltrange else 0);
		ax.scatter(horizontal, vertical, marker='o', s=siz, c=[shade[n] for n in range(31346)], edgecolors='none')
#		ax.scatter(horizontal, vertical, marker='o', s=siz, c=[(shade[n],shade[n],shade[n],transp) for n in range(31346)], edgecolors='none')
		ax.set_xlim(ranges[projh]); ax.set_ylim(ranges[projv])
		ax.set_xlabel(projh+projv+'-axis',fontsize=8, color=(.5,.5,.5)); ax.set_xticks([]); ax.set_yticks([])
#		plt.text(1,0, 'Stimulus '+str(stim), fontsize=12, ha='center', va='top', transform=axA.transAxes)	
	plt.savefig('viz_{:04d}.png'.format(t),dpi=120)


##
## Functions that work with deprecated simulation class. Need to be fixed
##


# Output plot
def make_plot(simulation, title, plot_simplices):
	# Set up 
	fig = plt.figure(figsize=(15,6))
	fig.suptitle(title, fontsize=18, y=.97)
	colorscheme = {'stimulus': plt.cm.autumn_r, 'voltage': plt.cm.Spectral_r, 'spike': (0.8353, 0.0, 0.1961)}

	# Define axes
	ax_stim = fig.add_subplot(3,1,1)
	ax_spikes = fig.add_subplot(3,1,2)
	ax_volts = fig.add_subplot(3,1,3)

	# Draw axis: stimulus
	ax_stim.invert_yaxis()
	stim_loc = []; stim_time = []; stim_power = []
	stim_range = [min([fire[3] for fire in simulation.stimulus]), max([fire[3] for fire in simulation.stimulus])]
	for fire in simulation.stimulus:
		for pulse in range(int(10*(fire[2]-fire[1]))):
			stim_loc.append(int(fire[0]))
			stim_time.append(fire[1]+pulse*0.1)
			stim_power.append((fire[3]-stim_range[0])/(stim_range[1]-stim_range[0]) if stim_range[1]!=stim_range[0] else fire[3])
	ax_stim.scatter(stim_time, stim_loc, s=0.5, marker="s", c=[colorscheme['stimulus'](p) for p in stim_power], edgecolors='none', alpha=0.8)	
	ax_stim.set_ylabel('thalamus index')
	ax_stim.set_xlim(0,simulation.length)

	# Draw axis: spike trains
	ax_spikes.invert_yaxis()
	ax_spikes.scatter(simulation.spikes['times'], simulation.spikes['senders'], s=0.5, marker="s", c=[colorscheme['spike'] for i in simulation.spikes['times']], edgecolors='none', alpha=0.8)
	ax_spikes.set_ylabel('neuron index')
	ax_spikes.set_ylim(1,simulation.neurons)
	ax_spikes.set_xlim(0,simulation.length)

	# Draw axis: voltage
	v = ax_volts.imshow(np.transpose(np.array(simulation.voltage).reshape(int(simulation.length*10-1),simulation.neurons)), cmap=colorscheme['voltage'], interpolation='None', aspect="auto")
	ax_volts.invert_yaxis()
	ax_volts.set_ylabel('neuron index')
	ax_volts.set_xlabel('time in ms')

	# Export figure	outlook
	fig.subplots_adjust(hspace=0.15, left=0.06, right=.96, bottom=0.05, top=.90)
	fig.align_ylabels([ax_stim,ax_spikes,ax_volts])
	fig.colorbar(v, ax=ax_volts, pad=0.02, fraction=0.01, orientation='vertical', label="voltage")
	plt.savefig('report_'+simulation.id+'.png')

	# Plot simplices if necessary
	if plot_simplices:
		plt.clf()
		fig = plt.figure(figsize=(8,6))
		plt.yscale("symlog")
		for dim in range(len(simulation.simplices[0])-1):
			plt.plot(range(len(simulation.simplices)), [simulation.simplices[step][dim] for step in range(len(simulation.simplices))], label='dim'+str(dim+1))
		plt.legend(bbox_to_anchor=(1, .5), loc='center left', borderaxespad=0.5)
		plt.xticks(list(range(len(simulation.simplices))))
		plt.ylabel('number of simplices')
		plt.savefig('simplexcount_'+simulation.id+'.png')

# Output voltage only plot
def make_volt_plot(simulation):
	# Set up 
	fig = plt.figure(figsize=(15,6))
	colorscheme = {'stimulus': plt.cm.autumn_r, 'voltage': plt.cm.Spectral_r, 'spike': (0.8353, 0.0, 0.1961)}

	# Draw axis: voltage
	plt.imshow(np.transpose(np.array(simulation.voltage).reshape(int(simulation.length-1),simulation.neurons)), cmap=colorscheme['voltage'], interpolation='None', aspect="auto")
	plt.gca().invert_yaxis()
	plt.ylabel('neuron index')
	plt.xlabel('time in ms')
	plt.colorbar(fraction=.1, pad=.01, orientation='vertical', label="voltage")
	plt.tight_layout()
	plt.savefig('voltage_'+simulation.id+'.png', dpi=200)

# Transimssion response matrices and simplex count
def make_tr(simulation, t1, t2, flagser):
	# Get key times
	times = [t1*i for i in range(int(simulation.length/t1)+1)]

	# Get ordered times and indices of spikes
	tr_times = sorted(simulation.spikes['times'])
	tr_neurons = [x-1 for _,x in sorted(zip(simulation.spikes['times'], simulation.spikes['senders']))]

	# Get adjacencies as dense matrix
	adjmat = scipy.sparse.coo_matrix((simulation.adj['data'], (simulation.adj['row'],simulation.adj['col'])), shape=(simulation.neurons,simulation.neurons)).toarray()
	matrices = []
	simplices = []
	for i in range(len(times)-1):
		print('    Step '+str(i)+': ['+str(times[i])+','+str(times[i+1])+'] in ['+str(times[i])+','+str(min(times[i]+t2,simulation.length))+']',flush=True)
		M = np.zeros((simulation.neurons,simulation.neurons),dtype='int8')
		t1_start = np.searchsorted(tr_times, times[i])
		t1_end = np.searchsorted(tr_times, times[i+1])
		t2_end = np.searchsorted(tr_times, times[i]+t2)

		# Source vertex spiked in [0, t1], sink in [0,t2]
		sources = np.unique(tr_neurons[t1_start:t1_end])
		targets = np.unique(tr_neurons[t1_start:t2_end])
		target_vector = scipy.sparse.coo_matrix((np.ones(len(targets),dtype='int8'), (np.zeros(len(targets),dtype='int8'), targets)), shape=(1,simulation.neurons+1)).toarray()[0][1:]
		for source in sources:
			M[source] = np.logical_and(adjmat[source],target_vector)
		
		# Source vertex spiked in [t1,t2], sink in [t1,t2]
		sources = np.unique(tr_neurons[t1_end:t2_end])
		target_vector = scipy.sparse.coo_matrix((np.ones(len(sources),dtype='int8'), (np.zeros(len(sources),dtype='int8'), sources)), shape=(1,simulation.neurons+1)).toarray()[0][1:]
		for source in sources:
			M[source] = np.logical_and(adjmat[source],target_vector)
		matrices.append(M)

		# Count simplices
		f = open('step'+str(i),'w')
		f.write('dim 0\n')
		for j in range(simulation.neurons):
			f.write('0 ')
		f.write('\ndim 1\n')
		for j in range(simulation.neurons):
			for k in np.nonzero(matrices[i][j])[0]:
				f.write(str(j)+' '+str(k)+'\n')
		f.close()
		cmd = subprocess.Popen(['./'+str(flagser), '--out', 'step'+str(i)+'.flag', 'step'+str(i)], stdout=subprocess.DEVNULL)
		cmd.wait()
		g = open('step'+str(i)+'.flag','r')
		L = g.readlines()
		g.close()
		simplices.append(np.array(list(map(lambda x: int(x), L[1][:-1].split(' ')[1:]))))
		print('    Simplex counts > 1: '+reduce((lambda x,y: str(x)+' '+str(y)), simplices[-1]), flush=True)
		subprocess.Popen(['rm', 'step'+str(i)], stdout=subprocess.DEVNULL)
		subprocess.Popen(['rm', 'step'+str(i)+'.flag'], stdout=subprocess.DEVNULL)

	# Save TR matrix
	print('    Compressing '+str(len(matrices))+' dense matrices into npz file', flush=True)
	np.savez_compressed('transmissionresponse_'+simulation.id, *matrices)

	# Save simplex list
	maxlen = max([len(s) for s in simplices])
	for s in range(len(simplices)):
		while len(simplices[s]) < maxlen:
			simplices[s] = np.concatenate((simplices[s],np.zeros(1,dtype='int64')), axis=None)
	np.save('simplices_'+simulation.id, np.array(simplices))
	return simplices

# Output h5 file of spike trains
def make_spikes(simulation):
	f = h5py.File('spikes_'+simulation.id+'.h5','w')
	spikes = simulation.spikes
	for k in spikes.keys():
		f.create_dataset(k, data=spikes[k])
	f.close()


