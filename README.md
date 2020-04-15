# neurotop-nest
Simplified Blue Brain Project reconstructions and validations

## About 

This project describes a construction and execution of a biological neural network model. The model is based on the EPFL-BlueBrain <a href="https://bbp.epfl.ch/nmc-portal/downloads">neocortical microcircuit</a>, though the setup allows for different models to be used.  

The main goals are for this project to be:
+ open-source,
+ executable on a personal computer,
+ easily modifiable.

## Setup

We will assume you are working in an Anaconda environment. First install NEST simulator and the relevant Python packages:

	conda install -c conda-forge nest-simulator 
	conda install numpy pandas pickle

Next, download the main Python file and structure / stimuli from the Dropbox folder. You only need the `nest_mc2.py` file for everything to work, but the other files are necessary for more complex constructions and experiments. Your folder should look like:

	nest_mc2.py
	structure/adjmat_mc2.npz
	strcuture/distances_mc2.npz
	strcuture/failures_mc2.pkl
	structure/synapses_mc2.pkl
	stimuli/fibres.npy

## Running an experiment

todo
