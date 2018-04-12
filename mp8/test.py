""" simple unit test for students """


import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
import sys
from k_means import k_means


C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.centers = k_means(C)

    def test_shape(self):
        result = self.centers
        self.assertEqual(result.shape, (3, 4))

    def test_no_override_arg(self):
        result = self.centers
        self.assertFalse(np.array_equal(C, result))


if __name__ == '__main__':
    unittest.main()
