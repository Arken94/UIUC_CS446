"""Simple unit tests for students."""

import numpy as np
import unittest
import tensorflow as tf
from run_computation import run_computation
from toy_functions import *


class RunComputationTests(unittest.TestCase):
    def test_run_simple(self):
        val = tf.constant(True)
        result = run_computation(val)
        self.assertEqual(result, True)

    def test_run_numerical(self):
        val = tf.constant(3)
        result = run_computation(val)
        self.assertEqual(result, 3)

        val = tf.constant(5.)
        result = run_computation(val)
        self.assertEqual(result, 5.)

    def test_array(self):
        val = tf.constant([1, 2])
        result = run_computation(val)
        np.testing.assert_array_equal(result, [1, 2])

    def test_complex(self):
        a = tf.Variable(5, dtype=tf.float32)
        b = tf.Variable(6, dtype=tf.float32)
        add = a + b
        result = run_computation(add)
        self.assertEqual(result, 11)


class RunToyFnTests(unittest.TestCase):
    def test_fn_1(self):
        arg1 = tf.constant([1, 4])
        arg2 = tf.constant([2, -1])
        correct = tf.constant([7, 28])
        attempt = toy_fn_1(arg1, arg2)
        result = run_computation(tf.reduce_all(tf.equal(correct, attempt)))
        self.assertEqual(result, True)

    def test_fn_1_rank2(self):
        arg1 = tf.constant([[1, 4], [5, 6]])
        arg2 = tf.constant([[2, -1], [9, 1]])
        correct = tf.constant([[7, 28], [399, 160]])
        attempt = toy_fn_1(arg1, arg2)
        result = run_computation(tf.reduce_all(tf.equal(correct, attempt)))
        self.assertEqual(result, True)

    def test_fn_2(self):
        arg1 = tf.constant([[1, 2], [3, 4]])
        arg2 = tf.constant([4, 2])
        correct = tf.constant([-1, 3])
        attempt = toy_fn_2(arg1, arg2)
        result = run_computation(tf.reduce_all(tf.equal(correct, attempt)))
        self.assertEqual(result, True)

    def test_fn_3(self):
        arg1 = tf.constant([1, 2])
        arg2 = tf.constant([10, 20])
        correct = tf.constant([1, 10, 2, 20])
        attempt = toy_fn_3(arg1, arg2)
        result = run_computation(tf.reduce_all(tf.equal(correct, attempt)))
        self.assertEqual(result, True)


if __name__ == '__main__':
    unittest.main()
