import numpy as np


def load_volts(voltage_path):
    volts_array = np.load(voltage_path, allow_pickle=True)
    volts = {'senders': volts_array[0], 'times': volts_array[1], 'V_m': volts_array[2]}
    return volts


def _volt_array(volt_dictionary, nnum):
    array = []
    for j in range(nnum):
        v_ind = np.where(volt_dictionary['senders'] == j+1)[0]
        array.append(volt_dictionary['V_m'][v_ind])
    return np.stack(array)


def load_spikes(spike_path):
    spikes_array = np.load(spike_path,
                           allow_pickle=True
                           )
    spikes = {'senders': spikes_array[0], 'times': spikes_array[1]}
    return spikes


def _spike_trains(spikes_dictionary, nnum, binsize, simlength):
    strains = []
    for j in range(nnum):
        try:
            sp_ind = np.where(spikes_dictionary['senders'] == j+1)[0]
            times = np.array(spikes_dictionary['times'][sp_ind])
            st = [np.count_nonzero(np.logical_and(times < node + binsize, times > node))
                  for node in list(range(0, simlength, binsize))]
        except TypeError:  # Neuron had no spikes
            st = [0 for _ in list(range(0, simlength, binsize))]
        strains.append(st)
    return np.stack(strains)
