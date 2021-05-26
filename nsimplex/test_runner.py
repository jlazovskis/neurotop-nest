from nsimplex_sim_runner import build_matrices, triu_from_array
from unittest import TestCase
from pathlib import Path
import numpy as np

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
    def test_data_save(self):
        pass

    def test_spikes_exist(self):
        pass

    def test_arguments_give_different_result(self):
        pass