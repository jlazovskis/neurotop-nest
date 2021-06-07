from unittest import TestCase
import numpy as np

from reliability_measures import gaussian_reliability, delayed_reliability


class TestGaussianReliability(TestCase):
    def test_reliable(self):
        spike_trains = [np.random.normal(1, size = 500).reshape((5,100))] * 10
        np.testing.assert_almost_equal(gaussian_reliability(spike_trains,1), np.array([1.,1.,1.,1.,1.]))

    def test_indipendence(self):
        spike_trains = [np.random.normal(1, size = 500).reshape((5,100)) for i in range(10)]
        np.testing.assert_array_less(gaussian_reliability(spike_trains,1), np.array([1.,1.,1.,1.,1.]))

    def test_smoothing(self):
        spike_trains = [np.random.normal(1, size = 500).reshape((5,100)) for i in range(10)]
        np.testing.assert_array_less(gaussian_reliability(spike_trains,1), gaussian_reliability(spike_trains,4))


class TestDelayedReliability(TestCase):
    def test_reliable(self):
        spike_trains = [np.random.normal(1, size = 500).reshape((5,100))] * 10
        np.testing.assert_almost_equal(delayed_reliability(spike_trains,0), np.array([1.,1.,1.,1.,1.]))

    def test_indipendence(self):
        spike_trains = [np.random.normal(1, size = 500).reshape((5,100)) for i in range(10)]
        np.testing.assert_array_less(delayed_reliability(spike_trains,2), np.array([1.,1.,1.,1.,1.]))
