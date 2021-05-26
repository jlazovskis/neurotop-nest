from nsimplex_sim_runner import build_matrices, triu_from_array, run_simulations
from unittest import TestCase
from pathlib import Path
import numpy as np
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

class TestSimulations(TestCase):
    @classmethod
    def setUpClass(cls):
        args = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplex', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 40, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit': 1, 'seed':0}
        run_simulations(Namespace(**args))

    #@classmethod
    #def tearDownClass(cls):
    #    structure_teardown_path = Path("structure/test")
    #    sim_teardown_path = Path("simulations/test")
    #    for file in structure_teardown_path.glob("**/*"):
    #        file.unlink()
    #    for file in sim_teardown_path.glob("**/*"):
    #        file.unlink()

    def test_arguments_give_different_result(self):
        args1 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplex1', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 20, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':1, 'seed':0}
        run_simulations(Namespace(**args1))
        for file1, file2 in zip(sorted(list(Path("test/3simplex1").glob("*"))), sorted(list(Path("test/3simplex").glob("*")))):
            self.assertNotEqual(md5(file1.open('rb').read()), md5(file2.open('rb').read()))

    def test_argload(self):
        args1 = pickle.load(Path("simulations/test/3simplexargs.pkl").open('rb'))
        run_simulations(Namespace(**args1))
        for file1, file2 in zip(sorted(list(Path("test/3simplex2").glob("*"))), sorted(list(Path("test/3simplex").glob("*")))):
            self.assertEqual(md5(file1.open('rb').read()), md5(file2.open('rb').read()))

    def test_seeds(self):
        args1 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexSeed0', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 20, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':0.8, 'seed':0}
        args2 = {'n': 3, 'root': '.', 'structure_path': 'test/3simplex', 'save_path': 'test/3simplexSeed1', 'stimulus_targets': 'all', 'stimulus_type': 'dc', 'stimulus_frequency': 1.0, 'noise_strength': 0.0, 'stimulus_strength': 20, 'stimulus_length': 90, 'stimulus_start': 5, 'time': 100, 'threads': 40, 'p_transmit':0.8, 'seed':1}
        run_simulations(Namespace(**args1))
        run_simulations(Namespace(**args2))
        for file1, file2 in zip(sorted(list(Path("test/3simplexSeed0").glob("*"))), sorted(list(Path("test/3simplexSeed1").glob("*")))):
            self.assertNotEqual(md5(file1.open('rb').read()), md5(file2.open('rb').read()))
        args3 = args1
        args3['save_path'] = "test/3simplexSeed2"
        run_simulations(Namespace(**args3))
        for file1, file2 in zip(sorted(list(Path("test/3simplexSeed0").glob("*"))), sorted(list(Path("test/3simplexSeed2").glob("*")))):
            self.assertEqual(md5(file1.open('rb').read()), md5(file2.open('rb').read()))

