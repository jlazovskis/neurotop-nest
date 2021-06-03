from nsimplex_sim_runner import build_matrices, triu_from_array, run_simulations, column_names
from nsimplex_sim_runner import load_volts, _volt_array, load_spikes, _spike_trains
from unittest import TestCase
from pathlib import Path
import numpy as np
import pandas as pd
from types import SimpleNamespace as Namespace
from hashlib import md5
import pickle


class TestAdjacencyMatrices(TestCase):
    @classmethod
    def tearDownClass(cls):
        teardown_path = Path("structure/test")
        for file in teardown_path.glob("**/*"):
            file.unlink()

    def test_signature(self):
        paths = build_matrices(Path("test/4simplex"), 4)
        for path in paths:
            matrix_from_save = np.load(Path("structure") / path)
            identifier = path.stem[-6:]
            id_array = np.array([int(e) for e in identifier])
            triu_part = triu_from_array(id_array, 4)
            matrix_from_identifier = np.triu(np.ones((4,4)), 1) + triu_part.T
            self.assertTrue(np.all(matrix_from_save == matrix_from_identifier))

    def test_triu_creations(self):
        array1 = np.array([1,2,3,4,5,6])
        triu_result = np.array([
                            [0, 1, 2, 3],
                            [0, 0, 4, 5],
                            [0, 0, 0, 6],
                            [0, 0, 0, 0],
                           ])
        self.assertTrue(np.all(triu_result == triu_from_array(array1, 4)))

def get_voltage_files(save_path):
    return sorted(list((Path('simulations') / save_path.parent).glob(save_path.name + "*volts.npy")))

def compare_voltage_traces(args1, args2):
    vt1 = get_voltage_files(Path(args1['save_path']))
    vt2 = get_voltage_files(Path(args2['save_path']))
    if not len(vt1) or not len(vt2):
        raise FileNotFoundError
    for file1, file2 in zip(vt1, vt2):
        print(file1)
        print(file2)
        if not np.all(_volt_array(load_volts(file1), args1['n']) == _volt_array(load_volts(file2), args2['n'])):
            return False
    return True


class TestSimulations(TestCase):
    @classmethod
    def tearDownClass(cls):
        structure_teardown_path = Path("structure/test")
        sim_teardown_path = Path("simulations/test")
        for file in structure_teardown_path.glob("**/*"):
            file.unlink()
        for file in sim_teardown_path.glob("**/*"):
            file.unlink()

    def test_arguments_give_different_result(self):
        args1 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexA', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 40, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit': 1, 'seed':0, 'binsize':3}
        args2 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexB', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 20, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':1, 'seed':1, 'binsize':3}
        run_simulations(Namespace(**args1))
        run_simulations(Namespace(**args2))
        self.assertFalse(compare_voltage_traces(args1, args2))

    def test_argload(self):
        args1 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexA', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 40, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit': 1, 'seed':0, 'binsize':3}
        run_simulations(Namespace(**args1))
        args2 = pickle.load(Path("simulations/test/3simplexAargs.pkl").open('rb'))
        args2['save_path'] = 'test/3simplexC'
        run_simulations(Namespace(**args2))
        self.assertTrue(compare_voltage_traces(args1, args2))

    def test_seeds(self):
        args1 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexSeedA', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 20, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':0.8, 'seed':1, 'binsize':3}
        args2 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexSeedB', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 20, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':0.8, 'seed':2, 'binsize':3}
        run_simulations(Namespace(**args1))
        run_simulations(Namespace(**args2))
        args3 = args1.copy()
        args3['save_path'] = "test/3simplexSeedC"
        run_simulations(Namespace(**args3))
        self.assertTrue(compare_voltage_traces(args1,args3))
        self.assertFalse(compare_voltage_traces(args1, args2))

    def test_noise(self):
        args1 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexNoiseA', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 1.0, 'stimulus_strength': 20, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':1, 'seed':1, 'binsize':3}
        args2 = args1.copy()
        args2['seed'] = 2
        args2['save_path'] = 'test/3simplexNoiseB'
        run_simulations(Namespace(**args1))
        run_simulations(Namespace(**args2))
        self.assertFalse(compare_voltage_traces(args1, args2))

    def test_poisson_st(self):
        args1 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexPPA', 'stimulus_targets': 'all', 'stimulus_type': 'poisson_parrot', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 10000, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':1, 'seed':1, 'binsize':3}
        args2 = args1.copy()
        args2['seed'] = 2
        args2['save_path'] = 'test/3simplexPPB'
        run_simulations(Namespace(**args1))
        run_simulations(Namespace(**args2))
        self.assertFalse(compare_voltage_traces(args1, args2))

    def test_bernoulli(self):
        args1 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexBSA', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 10, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':0.8, 'seed':1, 'binsize':3}
        args2 = args1.copy()
        args2['seed'] = 2
        args2['save_path'] = 'test/3simplexBSB'
        run_simulations(Namespace(**args1))
        run_simulations(Namespace(**args2))
        self.assertFalse(compare_voltage_traces(args1, args2))

    def test_dataframe(self):
        args1 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexDFA', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 10, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':0.8, 'seed':1, 'binsize':3}
        df1 = run_simulations(Namespace(**args1))
        self.assertEqual(df1.shape, (8,24))
        df2 = pd.read_pickle(Path('simulations') / (args1['save_path'] + 'dataframe.pkl'))
        self.assertTrue(df2.equals(df1))


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
        result = np.zeros((5,10))
        result[0, 0] = 1
        result[4, 9] = 1
        np.testing.assert_equal(sts, result)
