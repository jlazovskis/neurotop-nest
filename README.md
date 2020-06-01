# neurotop-nest
Simplified Blue Brain Project reconstructions and validations

## About 

This project describes a construction and execution of a biological neural network model. The model is based on the EPFL-BlueBrain <a href="https://bbp.epfl.ch/nmc-portal/downloads">neocortical microcircuit</a>, though the setup allows for different models to be used. Results of experiments run with this code are available at the <a href="https://homepages.abdn.ac.uk/neurotopology/data_portal/nest/">Neuro-Topology group website</a>. This project is related to several other projects:
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

Next, clone this project. You only need the `nest_mc2.py` file for everything to work, but the other files are necessary for more complex constructions and experiments.

## Running an experiment

Call the following command:

	python nest_mc2.py

This will run a simulation for 100ms on the 31346-neuron mc2 Blue Brain column, with lots of strong stimuli sent to the circuit. A spike file `spikes.npy` will be output. To make a visual overview of the experiment, call:

	python
	exec(open('next_mc2_output.py').read())
	make_spikeplot('spikes',100)
	make_tr_fromspikes('spikes',100,5,10)
	flag_tr('spikes')
	make_betticurves('spikes')

This will also compute the transmission response of the experiment for `t1=5` and `t2=10`, compute the homology of the flagged active graph, and plot pairs of betti curves against each other. Other functions to make outputs are in the file `nest_mc2_output.py`.


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
