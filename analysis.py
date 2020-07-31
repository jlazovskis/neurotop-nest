# neurotop-nest
#
# Commands for analysing NEST spike and volt output

# Packages
#import h5py                                                                 # For exporting h5 files
import numpy as np                                                          # For reading of files
import pandas as pd                                                         # For reading of files
from functools import reduce                                                # For combining simplices
import matplotlib as mpl                                                    # For plotting
import matplotlib.pyplot as plt                                             # For plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable                     # For plotting, to add colorbar nicely
import scipy.sparse                                                         # For exporting transimssion response matrices
import subprocess                                                           # For counting simplices and running flagser
from itertools import combinations                                          # For accessing specific plots
import os                                                                   # For deleting files
import argparse                                                             # For options

# Read arguments
parser = argparse.ArgumentParser(
	description='Simplified Blue Brain Project reconstructions and validations',
	usage='python analysis.py')
parser.add_argument('--spikes', type=str, help='Simulation name.')
parser.add_argument('--time', type=int, default=300, help='Length, in milliseconds, of experiment. Must be an integer. Default is 300.')
parser.add_argument('--circuit', type=str, default='drosophila', help='Either drosophila(default) or bbmc2.')
parser.add_argument('--skip_spikeplot', action='store_true', help='If included, creates raster plot of spikes.')
parser.add_argument('--TR', action='store_true', help='If included, creates transmission response graphs.')
parser.add_argument('--Betti_curves', action='store_true', help='If included, creates plot of Betti numbers.')
parser.add_argument('--flagser', type=str, default='/uoa/scratch/shared/mathematics/neurotopology/flagser', help='The address of flagser, needed if TR is used. Default is address on Maxwell.')
parser.add_argument('--t1', type=int, default=5, help='t1 paramater for transmission response. Default is 5ms.')
parser.add_argument('--t2', type=int, default=10, help='t2 paramater for transmission response. Default is 10ms.')
args = parser.parse_args()

# Output spike only plot
def make_spikeplot(name,length,step=5,circuit='bbmc2',recognize_inh=True):

	# Set styles and lists
	c_exc = (0.8353, 0.0, 0.1961)
	c_inh = (0.0, 0.1176, 0.3843)
	spike_file = np.load(circuit+'/simulations/'+name+'.npy',allow_pickle=True)[True][0]
	if recognize_inh:
		c_1 = c_exc if circuit == 'bbmc2' else c_inh
		c_0 = c_inh if circuit == 'bbmc2' else c_exc
		type_list = np.load(circuit+'/structure/bbmc2_excitatory.npy') if circuit == 'bbmc2' else np.load(circuit+'/structure/drosophila_inhibitory.npy')
		type_length = len(np.nonzero(type_list)[0])
		color_list = [c_1 if type_list[sender-1] else c_0 for sender in spike_file['senders']]
	else:
		color_list = [c_exc for sender in spike_file['senders']]

	# Set up figure
	fig, ax_spikes = plt.subplots(figsize=(20,6))
	plt.xlim(0,length)
	ax_spikes.set_ylim(0,nnum); ax_spikes.invert_yaxis()
	ax_percentage = ax_spikes.twinx(); ax_percentage.set_ylim(0,1)

	# Plot individual spikes
	ax_spikes.scatter(spike_file['times'], spike_file['senders'], s=1, marker="s", c=color_list, edgecolors='none', alpha=.8)
	ax_spikes.set_ylabel('neuron index')

	# Plot how much is spiking
	step_num = length//step
	times_by_timebin = [np.where(abs(np.array(spike_file['times'])-(t+.5)*step) < step/2)[0] for t in range(step_num)]
	type1_by_timebin = [[0]*nnum for t in range(step_num)]
	if recognize_inh:
		type0_by_timebin = [[0]*nnum for t in range(step_num)]
		for t in range(step_num):
			for sender in np.unique(np.array(spike_file['senders'])[times_by_timebin[t]]):
				type1_by_timebin[t][sender-1] = type_list[sender-1]
				type0_by_timebin[t][sender-1] = not type_list[sender-1]
		ax_percentage.step([t*step for t in range(1,step_num+1)], [np.count_nonzero(np.array(timebin))/type_length for timebin in type1_by_timebin], color=c_1)
		ax_percentage.step([t*step for t in range(1,step_num+1)], [np.count_nonzero(np.array(timebin))/(nnum-type_length) for timebin in type0_by_timebin], color=c_0)
	else:
		for t in range(step_num):
			for sender in np.unique(np.array(spike_file['senders'])[times_by_timebin[t]]):
				type1_by_timebin[t][sender-1] = 1
		ax_percentage.step([t*step for t in range(1,step_num+1)], [np.count_nonzero(np.array(timebin))/nnum for timebin in type1_by_timebin], color=c_exc)

	ax_percentage.set_ylabel('percentage spiking')

	# Format and save figure
	plt.tight_layout()
	plt.savefig(circuit+'/simulations/'+name+'.png', dpi=200)


# Transimssion response matrices from spike file
def make_tr_fromspikes(name, length, t1, t2):
	s = np.load('./bbmc2/simulations/'+name+'.npy',allow_pickle=True)[True][0]
	adj = scipy.sparse.load_npz('./bbmc2/structure/adjmat_mc2.npz').toarray()
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
def flag_tr(name):
	tr = np.load('./bbmc2/simulations/'+name+'.npz')
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
		if os.path.exists('step'+str(step)+'.out'):
			os.remove('step'+str(step)+'.out')
		cmd = subprocess.Popen([flagser, '--out', 'step'+str(step)+'.out', 'step'+str(step)+'.in'], stdout=subprocess.DEVNULL); cmd.wait()
		g = open('step'+str(step)+'.out','r'); L = g.readlines(); g.close()
		print('Step '+str(step)+': '+str(list(map(lambda x: int(x), L[1][:-1].split(' ')[1:]))),flush=True)
		bettis.append(np.array(list(map(lambda x: int(x), L[1][:-1].split(' ')[1:]))))
		# Remove files
		subprocess.Popen(['rm', 'step'+str(step)+'.in'], stdout=subprocess.DEVNULL)
		subprocess.Popen(['rm', 'step'+str(step)+'.out'], stdout=subprocess.DEVNULL)
	np.save(name+'_TR.npy',np.array(bettis))

# Make betti curves from flagged files
def make_betticurves(name,params={'start':0, 'end':250, 'step':5}):
	# Load file and make dim lists
	bettis_bystep = np.load('./bbmc2/simulations/'+name+'_TR.npy',allow_pickle=True)
	#stepnum = len(bettis_bystep)
	stepfirst = params['start']//params['step']
	steplast = params['end']//params['step']
	stepnum = steplast-stepfirst-1
	bettis_bydim = {i:[] for i in range(1,5)}
	for i in range(1,5):
		for step in range(len(bettis_bystep)):
			bettis_bydim[i].append(bettis_bystep[step][i-1] if len(bettis_bystep[step]) > i else 0)
	linecolors = [plt.get_cmap('hsv')(i/(stepnum-1)) for i in range(stepnum)]

	# Set up figure
	siz = 40
	fig = plt.figure(figsize=(18,4)) # default is (8,6)
	mpl.rcParams['axes.spines.right'] = False; mpl.rcParams['axes.spines.top'] = False

	# Make legend
	axLEG = fig.add_subplot(1,4,1)
	axLEG.set_xlim([0,2.1]); axLEG.set_ylim([0,1])
	axLEG.set_xticks([]); axLEG.set_yticks([])
	axLEG.axis('off')
	axLEG.scatter([.05+2*i/stepnum for i in range(stepnum+1)], [.5 for i in range(stepnum+1)], marker='o', s=1.5*siz, c=[plt.get_cmap('hsv')(i/stepnum) for i in range(stepnum+1)], zorder=1, alpha=.5, edgecolors='none')
	for i in range(stepnum):
		axLEG.plot([.05+i*2/stepnum,.05+(i+1)*2/stepnum], [.5,.5], c=linecolors[i], zorder=-1, linewidth=3, alpha=.5)
	for i in range(stepnum+1):
		plt.text(.05+i*2/stepnum, 0.45, str(params['step']*(stepfirst+i)), fontsize=10, horizontalalignment='center', verticalalignment='top')
	plt.text(.05, 0.3, 'Seconds along experiment', fontsize=12, horizontalalignment='left', verticalalignment='center')

	# Make betti curves
	fig.add_subplot(1,4,2); plt.xscale("symlog"); plt.yscale("symlog")
	fig.add_subplot(1,4,3); plt.xscale("symlog"); plt.yscale("symlog")
	fig.add_subplot(1,4,4); plt.xscale("symlog"); plt.yscale("symlog")
	axes = fig.axes[1:]
	for dim,ax in zip(combinations(range(1,4),2),axes):
		segments_x = [[bettis_bydim[dim[0]][i],bettis_bydim[dim[0]][i+1]] for i in range(stepfirst,steplast)]
		segments_y = [[bettis_bydim[dim[1]][i],bettis_bydim[dim[1]][i+1]] for i in range(stepfirst,steplast)]
		for i in range(stepnum):
			ax.plot(segments_x[i], segments_y[i], c=linecolors[i], zorder=-1, linewidth=3, alpha=.5)
		ax.scatter(bettis_bydim[dim[0]][stepfirst:steplast+1], bettis_bydim[dim[1]][stepfirst:steplast+1], marker='o', s=siz, c=[plt.get_cmap('hsv')(i/stepnum) for i in range(stepnum+2)], zorder=1, alpha=.5, edgecolors='none')
		ax.set_xlabel(r'$\beta_{0:g}$'.format(dim[0]), fontsize=12); ax.xaxis.set_label_coords(1.08, 0.04)
		ax.set_ylabel(r'$\beta_{0:g}$'.format(dim[1]), fontsize=12); ax.yaxis.set_label_coords(0.03, 1.05)

	# Export
	fig.subplots_adjust(wspace=0.3, hspace=0.45, right=.96, left=0.02, top=0.90, bottom=0.08)
	plt.savefig(name+'_betticurves.png',dpi=200)

# Combine two output pictures
def combine_spikes_bettis(name):
	widths = []
	for suffix in ['.png','_betticurves.png']:
		cmd = subprocess.Popen(['file',name+suffix],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
		out = cmd.communicate()[0].decode('utf-8').split('\n')[:-1]
		widths.append(int(out[0].split(' ')[4]))
	cmd = subprocess.Popen(['convert',name+'.png','-resize',str(min(widths)),'top.png'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); cmd.wait()
	cmd = subprocess.Popen(['convert',name+'_betticurves.png','-resize',str(min(widths)),'bottom.png'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); cmd.wait()
	cmd = subprocess.Popen(['convert','top.png','bottom.png','-append',name+'_spikes_bettis.png'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); cmd.wait()
	subprocess.Popen(['rm','top.png'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);
	subprocess.Popen(['rm','bottom.png'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);


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
	nnum = {'bbmc2': 31346, 'drosophila': 25288}[args.circuit]

	if not args.skip_spikeplot:
	    make_spikeplot(args.spikes,args.time,step=5,circuit=args.circuit,recognize_inh=(args.circuit=='bbmc2'))

	if args.TR:
	    make_tr_fromspikes(args.spikes,time,args.t1,args.t2)
	    flag_tr(args.spikes)

	if args.Betti_curves:
	    make_betticurves(args.spikes,{'start':10, 'end':95, 'step':5})
	    combine_spikes_bettis(args.spikes)
