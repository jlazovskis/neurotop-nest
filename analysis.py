# neurotop-nest
#
# Commands for analysing NEST spike and volt output

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
from itertools import combinations                                          # For accessing specific plots

# Output spike only plot
def make_spikeplot(name,length):
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
		cur_edges = []
		for spike_index in np.where(abs(s['times']-(step+.5)*t1) < t1/2)[0]:
			spike_time = s['times'][spike_index]
			spiker = s['senders'][spike_index]-1
			for alsospiked in np.intersect1d(np.nonzero(adj[spiker])[0], np.array([n-1 for n in s['senders'][np.where(abs(s['times']-spike_time-t2/2) < t2/2)[0]]])):
				cur_edges.append(tuple({spiker,alsospiked}))
		active_edges.append(list(set(cur_edges)))
		print('Time ['+str(step*t1)+','+str((step+1)*t1)+'] in ['+str(step*t1)+','+str(step*t1+t2)+'] has '+str(len(cur_edges))+' active edges', flush=True)
	print('Compressing edge lists into npz file', flush=True)
	np.savez_compressed(name+'.npz', *active_edges)


# Run flagser on transmission response edge lists
def flag_tr(name, flagser='/home/jlv/flagser/flagser'):
	tr = np.load(name+'.npz')
	bettis = []
	for step in range(len(tr)):
		cur_bettis = []
		# Make file for Flagser
		f = open('step'+str(step)+'.in','w'); f.write('dim 0\n')
		for i in range(nnum):
			f.write('0 ')
		f.write('\ndim 1\n')
		for edge in tr['arr_'+str(step)]:
			f.write(str(edge[0])+' '+str(edge[1])+'\n')
		f.close()
		# Run flagser and read output file
		cmd = subprocess.Popen([flagser, '--out', 'step'+str(step)+'.out', 'step'+str(step)+'.in'], stdout=subprocess.DEVNULL); cmd.wait()
		g = open('step'+str(step)+'.out','r'); L = g.readlines(); g.close()
		print('Step '+str(step)+': '+str(list(map(lambda x: int(x), L[1][:-1].split(' ')[1:]))),flush=True)
		bettis.append(np.array(list(map(lambda x: int(x), L[1][:-1].split(' ')[1:]))))
		# Remove files
		subprocess.Popen(['rm', 'step'+str(step)+'.in'], stdout=subprocess.DEVNULL)
		subprocess.Popen(['rm', 'step'+str(step)+'.out'], stdout=subprocess.DEVNULL)
	np.save(name+'_TR.npy',np.array(bettis))

# Make betti curves from flagged files
def make_betticurves(name):
	# Load file and make dim lists
	bettis_bystep = np.load(name+'_TR.npy',allow_pickle=True); stepnum = len(bettis_bystep)
	bettis_bydim = {i:[] for i in range(1,5)}
	for i in range(1,5):
		for step in range(stepnum):
			bettis_bydim[i].append(bettis_bystep[step][i-1] if len(bettis_bystep[step]) > i else 0)
	linecolors = [plt.get_cmap('hsv')(i/(stepnum-1)) for i in range(stepnum)]

	# Set up figure
	siz = 30
	fig = plt.figure(figsize=(12,10)) # default is (8,6)
	mpl.rcParams['axes.spines.right'] = False; mpl.rcParams['axes.spines.top'] = False

	# Make legend
	axLEG = fig.add_subplot(1,1,1)
	axLEG.set_xlim([0,3]); axLEG.set_ylim([0,3])
	axLEG.set_xticks([]); axLEG.set_yticks([])
	axLEG.axis('off')
	axLEG.scatter([.05+2*i/stepnum for i in range(stepnum)], [.5 for i in range(stepnum)], marker='o', s=siz, c=[plt.get_cmap('hsv')(i/(stepnum-1)) for i in range(stepnum)], zorder=1, alpha=.5)
	for i in range(stepnum-1):
		axLEG.plot([.05+i*2/stepnum,.05+(i+1)*2/stepnum], [.5,.5], c=linecolors[i], zorder=-1, linewidth=3, alpha=.5)
		if i%5 == 0:
			plt.text(.05+i*2/stepnum, 0.45, str(i), fontsize=10, horizontalalignment='center', verticalalignment='top')
	plt.text(.05, 0.3, 'Step along experiment', fontsize=12, horizontalalignment='left', verticalalignment='center')

	# Make betti curves
	for row in range(4):
		for col in range(row+1,4):
			fig.add_subplot(3,3,row*3+col); plt.xscale("symlog"); plt.yscale("symlog")
	axes = fig.axes[1:]
	for dim,ax in zip(combinations(range(1,5),2),axes):
		segments_x = [[bettis_bydim[dim[0]][i],bettis_bydim[dim[0]][i+1]] for i in range(stepnum-1)]
		segments_y = [[bettis_bydim[dim[1]][i],bettis_bydim[dim[1]][i+1]] for i in range(stepnum-1)]
		for i in range(stepnum-1):
			ax.plot(segments_x[i], segments_y[i], c=linecolors[i], zorder=-1, linewidth=3, alpha=.5)
		ax.scatter(bettis_bydim[dim[0]], bettis_bydim[dim[1]], marker='o', s=siz, c=[plt.get_cmap('hsv')(i/(stepnum-1)) for i in range(stepnum)], zorder=1, alpha=.5)
		ax.set_xlabel(r'$\beta_{0:g}$'.format(dim[0]), fontsize=12); ax.xaxis.set_label_coords(1.1, 0.05)
		ax.set_ylabel(r'$\beta_{0:g}$'.format(dim[1]), fontsize=12); ax.yaxis.set_label_coords(0.03, 1.1)

	# Export
	fig.subplots_adjust(wspace=0.4, hspace=0.45, right=.94, left=0.05, top=0.93, bottom=0.04)
	plt.savefig(name+'_betticurves.png',dpi=200)


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
## Run the desired commands
##

if __name__=="__main__":
	#    num: number of neurons in circuit
	# spikes: name of output spike file
	#   time: length of experiment
	#     t1: t1 paramater for transmission response
	#     t2: t2 paramater for transmission response
	nnum = 31346
	spikes = 'bbmc2_n15_1591291009.npy'
	time = 250
	t1 = 5
	t2 = 10
	# Functions to call
	make_spikeplot(spikes[:-4],time)
	make_tr_fromspikes(spikes[:-4],time,5,10)
	flag_tr(spikes[:-4])
	make_betticurves(spikes[:-4])