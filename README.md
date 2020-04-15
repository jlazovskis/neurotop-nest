# neurotop-nest
Simplified Blue Brain Project reconstructions and validations

## About 

This project describes a construction and execution of a biological neural network model. The model is based on the EPFL-BlueBrain <a href="https://bbp.epfl.ch/nmc-portal/downloads">neocortical microcircuit</a>, though the setup allows for different models to be used.

Results of experiments run with this code are available at the <a href="https://homepages.abdn.ac.uk/neurotopology/data_portal/nest/">Neuro-Topology group website</a>.

The main goals are for this project to be:
+ open-source,
+ executable on a personal computer,
+ easily modifiable.

## Setup

We will assume you are working in an Anaconda environment. First install NEST simulator and the relevant Python packages:

	conda install -c conda-forge nest-simulator 
	conda install numpy pandas pickle

Next, clone this project. You only need the `nest_mc2.py` file for everything to work, but the other files are necessary for more complex constructions and experiments.

## Running an experiment

Simply call:

	python nest_mc2.py

This will run a simulation for 100ms on the 31346-neuron mc2 Blue Brain column, with lots of strong stimuli sent to the circuit. A file `report.png` will be produced, giving a visual overview of the experiment.

## Benchmarks

todo
