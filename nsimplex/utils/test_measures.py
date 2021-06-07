from unittest import TestCase
import numpy as np

from uniformity_measures import (average_pearson, average_cosine_distance, spike_range, average_pearson_directional,
                                 pearson_range, pearson_matrix, spike_count)

# Uniformity measures tests


class CorrelationTest(TestCase):
    def test_average_pearson(self):
        array1 = np.array([
            [1,2,1,2],
            [2,1,2,1],
            [-1,-2,-1,-2],
            [-2,-1,-2, -1]
           ])
        self.assertAlmostEqual(average_pearson(array1), -1/3)
        array2 = np.ones((4,4))
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(np.isnan(average_pearson(array2)))
        array3 = np.array([
            [1,2,3,4,5],
            [-1,-2,-3,-4,-5]
        ])
        self.assertAlmostEqual(average_pearson(array3), -1.)

    def test_average_pearson_directional(self):
        array1 = np.random.rand(4,4)
        graph = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
           ])
        self.assertAlmostEqual(average_pearson(array1), average_pearson_directional(array1, graph))

    def test_pearson_matrix(self):
        array1 = np.array([
            [1,2,1,2],
            [2,1,2,1],
            [-1,-2,-1,-2],
            [-2,-1,-2, -1]
           ])
        result = np.array([
            [1., -1., -1., 1.],
            [-1., 1., 1., -1.],
            [-1., 1., 1., -1.],
            [1., -1., -1., 1.]
           ])
        self.assertTrue(np.all(pearson_matrix(array1)==result))

    def test_pearson_range(self):
        array1 = np.array([
            [1,2,1,2],
            [2,1,2,1],
            [-1,-2,-1,-2],
            [-2,-1,-2, -1]
           ])
        result = 2.
        self.assertAlmostEqual(pearson_range(array1), result)


class CosineDistanceTest(TestCase):
    def test_cosine_distance(self):
        array1 = np.array([
            [1,2,1,2],
            [2,1,2,1],
            [-1,-2,-1,-2],
            [-2,-1,-2, -1]
           ])
        self.assertAlmostEqual(average_cosine_distance(array1), -1/3)
        array2 = np.ones((4,4))
        self.assertAlmostEqual(average_cosine_distance(array2), 1.)
        array3 = np.array([
            [1,2,3,4,5],
            [-1,-2,-3,-4,-5]
        ])
        self.assertAlmostEqual(average_cosine_distance(array3), -1.)


class SpikeTrainRangeTest(TestCase):
    def test_st_range(self):
        array1 = np.array([
            [1,0,1,2],
            [2,1,2,1],
            [0,1,1,2],
            [2,1,2, 1]
           ])
        self.assertEqual(spike_range(array1), 2)
        array2 = np.ones((4,4))
        self.assertEqual(spike_range(array2), 0)
        array3 = np.array([
            [1,2,3,4,5],
            [2,3,4,5,6]
        ])
        self.assertEqual(spike_range(array3), 5)
