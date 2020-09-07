import matplotlib.pyplot as plt
import numpy as np

# Output spike only plot
def make_spikeplot(name,length,step=5):

	# Set styles and lists
	c_exc = (0.8353, 0.0, 0.1961)
	c_inh = (0.0, 0.1176, 0.3843)
	spike_file = np.load(name+'.npy')
	color_list = [c_exc for sender in spike_file[0]]

	# Set up figure
	fig, ax_spikes = plt.subplots(2,1,figsize=(20,6),sharex=True)
	ax_spikes[1].set_xlim(0,length);
	ax_spikes[0].set_ylim(0,nnum); ax_spikes[0].invert_yaxis()
	ax_percentage = ax_spikes[1]

	# Plot individual spikes
	ax_spikes[0].scatter(spike_file[1], spike_file[0], s=1, marker="s", c=color_list, edgecolors='none', alpha=.8)
	ax_spikes[0].set_ylabel('neuron index')

	# Plot how much is spiking
	step_num = length//step
	times_by_timebin = [np.where(abs(np.array(spike_file[1])-(t+.5)*step) < step/2)[0] for t in range(step_num)]
	type1_by_timebin = [[0]*nnum for t in range(step_num)]
	for t in range(step_num):
		for sender in np.unique(np.array(spike_file[0])[times_by_timebin[t]]):
			type1_by_timebin[t][int(sender-1)] = 1
	ax_percentage.step([t*step for t in range(1,step_num+1)], [np.count_nonzero(np.array(timebin))/nnum for timebin in type1_by_timebin], color=c_exc)

	ax_percentage.set_ylabel('percentage spiking')

	# Format and save figure
	plt.tight_layout()
	plt.savefig(name+'.png', dpi=200)
