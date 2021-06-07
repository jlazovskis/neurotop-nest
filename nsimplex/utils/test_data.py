import numpy as np
from unittest import TestCase

from data import load_volts, _volt_array, load_spikes, _spike_trains


class TestSpikeTrains(TestCase):
    def test_empty_spike_train(self):
        nnum = 5
        binsize = 10
        simlength = 100
        spike_dictionary = {'senders': np.array([]), 'times': np.array([])}
        sts = _spike_trains(spike_dictionary, nnum, binsize, simlength)
        self.assertTrue(not np.any(sts))
        self.assertEqual(sts.shape, (5,10))

    def test_spike_train(self):
        nnum = 5
        binsize = 10
        simlength = 100
        spike_dictionary = {'senders': np.array([1, 5]), 'times': np.array([5, 95])}
        sts = _spike_trains(spike_dictionary, nnum, binsize, simlength)
        result = np.zeros((5, 10))
        result[0, 0] = 1
        result[4, 9] = 1
        np.testing.assert_equal(sts, result)