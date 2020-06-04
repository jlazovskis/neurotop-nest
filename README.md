# neurotop-nest
Reconstructions and validations of neurological circuits for topological analysis

## About 

This project provides a method to work with networks as if they were biological networks. The primary use case is neurological networks, particularly EPFL-BlueBrain <a href="https://bbp.epfl.ch/nmc-portal/downloads">neocortical microcircuit</a>. Results of experiments run with this code are available at the <a href="https://homepages.abdn.ac.uk/neurotopology/data_portal/nest/">Neuro-Topology group website</a>. This project is related to several other projects:
+ <a href="https://github.com/nest/nest-simulator">NEST simulator</a> for the simulation environment
+ <a href="https://github.com/luetge/flagser">flagser</a> for counting active simplices during the simulation

The main goals are for this project to be:
+ open-source,
+ executable on a personal computer,
+ easily modifiable.

## Setup

We will assume you are working in an Anaconda environment. First install NEST simulator and the relevant Python packages:

	conda install -c conda-forge nest-simulator 
	conda install numpy scipy pandas pickle h5py

Next, clone this project. You only need the `bbmc2.py` file for everything to work, but the other files are necessary for more complex constructions and experiments.

## Running an experiment

Simply call `python bbmc2.py`. This will run a simulation for 100ms on the 31346-neuron mc2 Blue Brain column, with lots of strong stimuli sent to the circuit. A spike file with a name like`bbmc2_constant_1591291009.npy` will be output. To make a visual overview of the experiment, open the file `analysis.py` and paste the name of this output spike file, adjusting the otehr parameters as necessary. The end of the file should look like this:

	nnum = 31346
	spikes = 'bbmc2_n15_1591291009.npy'
	time = 250
	t1 = 5
	t2 = 10

Then call `python analysis.py` to analyze the output spike file. This will also compute the transmission response of the experiment for `t1=5` and `t2=10`, compute the homology of the flagged active graph, and plot pairs of betti curves against each other.

### Options

The following options can be given the Python call initianting the file, as `--option_name=option_value`. Default values of options are indicated below, which (along with descriptions) are in the main file.

| Option              | Type    | Default             | Description                                                                                                   |
| ------------------- | ------- | --------------------| ------------------------------------------------------------------------------------------------------------- |
| `--fibres`          | string  | `fibres.npy`        | Filename of fibres (must be in folder "stimuli"). List of lists of GIDs. Length is number of thalamic fibers. |
| `--stimulus`        | string  | `constant`          | Firing pattern of stimulus (must be in folder "stimuli"), must be one of "n5","n15","n30","constant".         |
| `--time`            | integer | `100`               | Length, in milliseconds, of experiment.                                                                       |


### Flags

The following flags can be given the Python call initianting the file, as `--flag_name`. Default behavior when flags are not called is indicated below.

| Flag              | If not called (default)                                                                             | If called                                            |
| ----------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `--no_mc2approx`  | Approximate structure (distances between neurons, weights / delays / failures of synapses) is used. | Nothing beyond mc2 neuron connections is used.       |
| `--shuffle`       | Adjacency matrix is used as given.                                                                  | Rows and columns of adjacency matrix are shuffled.   |


## Benchmarks

ToDo.

Takes about 2 minutes to run a 250 ms simulation. The line `nest.SetKernelStatus({"local_num_threads": 8})` indicates that the computer being used has 8 threads. This can (should) be adjusted for your own system.
