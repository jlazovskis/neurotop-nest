from unittest import TestCase
from uniformity_measures import (average_pearson, average_cosine_distance)
import numpy as np

# Uniformity measures tests

class AverageCorrelationTest(TestCase):
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
