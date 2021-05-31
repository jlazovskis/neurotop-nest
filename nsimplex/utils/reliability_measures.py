import numpy as np
import pandas as pd
from typing import List
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

# Independent function import
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
module_path = (Path(__file__).parent / "uniformity_measures.py").absolute()
module_name = "um"
um_spec = spec_from_file_location(module_name, str(module_path))
um = module_from_spec(um_spec)
um_spec.loader.exec_module(um)
#############################

def gaussian_reliability(spike_trains_list: List[np.ndarray], gaussian_variance: float):
    convolved_signals = np.stack([gaussian_filter1d(spike_trains, gaussian_variance) for spike_trains in spike_trains_list], axis = 0)
    print(convolved_signals)
    result = [um.average_cosine_distance(convolved_signals[:,i,:]) for i in range(convolved_signals.shape[1])]
    return np.array(result)

def delayed_reliability(spike_trains_list: List[np.ndarray], shift):
    result = []
    nsamples = len(spike_trains_list)
    for neuron in range(spike_trains_list[0].shape[0]):
        neuron_values = []
        for i in range(nsamples):
            for j in range(i+1, nsamples):
                neuron_values.append(cross_correlation_average(spike_trains_list[i][neuron,:], spike_trains_list[j][neuron,:], shift))
        result.append(np.mean(neuron_values))
    return np.array(result)

def cross_correlation_average(st1, st2, shift):
    values = []
    signal_length = len(st1)
    for i in range(-shift, shift+1):
        indices = np.mod(np.arange(i, signal_length+1), signal_length)
        values.append(correlation(st1[indices], st2[indices]))
    return np.mean(values)

def correlation(st1, st2):
    a = np.mean(np.multiply(st1 - np.mean(st1), st2 - np.mean(st2)))/np.std(st1)/np.std(st2)
    return a
