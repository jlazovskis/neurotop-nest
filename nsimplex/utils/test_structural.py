import numpy as np
from structural import (
    indegree_range,
    outdegree_range,
    bidegree_range,
    degree_range,
    directionality,
    maximal_simplex_count,
    bidirectional_edges
)
from unittest import TestCase


class TestRanges(TestCase):
    def setUp(self):
        self.array1 = np.array([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        self.array2 = np.array([
            [False, True, True, False],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        self.array3 = np.array([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, True, False],
        ])

    def test_indegree_range(self):
        self.assertEqual(indegree_range(self.array1), 3)
        self.assertEqual(indegree_range(self.array2), 2)
        self.assertEqual(indegree_range(self.array3), 3)

    def test_oudegree_range(self):
        self.assertEqual(outdegree_range(self.array1), 3)
        self.assertEqual(outdegree_range(self.array2), 2)
        self.assertEqual(outdegree_range(self.array3), 2)

    def test_degree_range(self):
        self.assertEqual(degree_range(self.array1), 0)
        self.assertEqual(degree_range(self.array2), 1)
        self.assertEqual(degree_range(self.array3), 1)

    def test_bidegree_range(self):
        self.assertEqual(bidegree_range(self.array1), 0)
        self.assertEqual(bidegree_range(self.array2), 0)
        self.assertEqual(bidegree_range(self.array3), 1)

class TestDirectionality(TestCase):
    def setUp(self):
        self.array1 = np.array([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        self.array2 = np.array([
            [False, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [True, True, True, False],
        ])
        self.array3 = np.array([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, True, False],
        ])

    def test_directionality(self):
        self.assertEqual(directionality(self.array1), 20)
        self.assertEqual(directionality(self.array2), 0)
        self.assertEqual(directionality(self.array3), 18)


class TestMaximalCount(TestCase):
    def setUp(self):
        self.array1 = np.array([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        self.array2 = np.array([
            [False, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [True, True, True, False],
        ])
        self.array3 = np.array([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, True, False],
        ])

    def testSimplexCount(self):
        self.assertEqual(maximal_simplex_count(self.array1), 1)
        self.assertEqual(maximal_simplex_count(self.array2), 24)
        self.assertEqual(maximal_simplex_count(self.array3), 2)

class TestBidirectionalEdges(TestCase):
    def setUp(self):
        self.array1 = np.array([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        self.array2 = np.array([
            [False, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [True, True, True, False],
        ])
        self.array3 = np.array([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, True, False],
        ])

    def testBidirectionalEdges(self):
        self.assertEqual(bidirectional_edges(self.array1), 0)
        self.assertEqual(bidirectional_edges(self.array2), 6)
        self.assertEqual(bidirectional_edges(self.array3), 1)
