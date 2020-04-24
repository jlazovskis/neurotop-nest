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

Simply call:

	python nest_mc2.py

This will run a simulation for 100ms on the 31346-neuron mc2 Blue Brain column, with lots of strong stimuli sent to the circuit. A file `report.png` will be produced, giving a visual overview of the experiment.

### Options

The following options can be given the Python call initianting the file, as `--option_name=option_value`. Default values of options are indicated below, which (along with descriptions) are in the main file.

| Option              | Type    | Default                                | Description                                                                                                   |
| ------------------- | ------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--fibres`          | string  | `fibres.npy`                           | Filename of fibres (must be in folder "stimuli"). List of lists of GIDs. Length is number of thalamic fibers. |
| `--stimulus`        | string  | `constant_firing.npy`                  | Filename of firing pattern of stimulus (must be in folder "stimuli"). List of tuples (index,start,stop,rate). |
| `--time`            | integer | `100`                                  | Length, in milliseconds, of experiment.                                                                       |
| `--outplottitle`    | string  | `Spikemeter and voltmeter reports`     | Title of the output plot.                                                                                     |
| `--t1`              | float   | `5.0`                                  | Transmission reponse: time for source to spike                                                                |
| `--t2`              | float   | `10.0`                                 | Transmission reponse: time for sink to spike                                                                  |
| `--flagser`         | string  | `../flagser/flagser`                   | Local location of flagser. Only relevant if `--count_simplices` flag is called.                               |


### Flags

The following flags can be given the Python call initianting the file, as `--flag_name`. Default behavior when flags are not called is indicated below.

| Flag                | If not called (default)                                                                               | If called                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `--no_mc2approx`    | Approximate structure (distances between neurons, weights / delays / failures of synapses) is used.   | Nothing beyond mc2 neuron connections is used.       |
| `--shuffle`         | Adjacency matrix is used as given.                                                                    | Rows and columns of adjacency matrix are shuffled.   |
| `--make_spikes`     | No spike trains are computed.                                                                         | Output `h5` file of the spiketrains.                 |
| `--make_tr`         | No transmission response matrices are computed.                                                       | Output `npz` file of transmission reponse matrices.  |
| `--count_simplices` | Simplices of transmission response time bins are not computed.                                        | Output `npy` file and plot of simplex counts.        |
| `--no_plot`         | A plot of the spiketrains and voltages is output.                                                     | No plot is output.                                   |


## Benchmarks

todo
