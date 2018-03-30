"""Simple unit tests for students."""

import unittest
import numpy as np
from linear_mrf import LinearMRF
import tensorflow as tf


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = LinearMRF(3, 2)

    def test_unary_feature_shape(self):
        img = np.array([1, 0, 0, 1, 0, 1])
        result = self.model.get_unary_features(img)
        self.assertEqual(result.shape, (6, 2))

    def test_unary_feature_data(self):
        img = np.array([1, 0, 0, 1, 0, 1])
        result = self.model.get_unary_features(img)
        expected = np.array([
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1]
        ])
        np.testing.assert_array_equal(result, expected)

    def test_pairwise_feature_shape(self):
        result = self.model.get_pairwise_features()
        self.assertEqual(result.shape, (7, 4))

    def test_pairwise_feature_data(self):
        result = self.model.get_pairwise_features()
        np.testing.assert_array_equal(result, [[1, 0, 0, 1]] * 7)


class BeliefTests(unittest.TestCase):
    def setUp(self):
        self.model = LinearMRF(1, 2)
    
    def test_belief_convergence(self):
        new_beliefs = np.array([[0, 1],
                                [0, 1]])
        old_beliefs = np.array([[1, 0],
                                [1, 0]])
        result = self.model.check_convergence(new_beliefs, old_beliefs, 0.2)
        self.assertEqual(result, False)

    def test_pairwise_beliefs_shape(self):
        beliefs = np.array([[0, 1],
                            [1, 0]])
        result = self.model.get_pairwise_beliefs(beliefs)
        self.assertEqual(result.shape, (1, 4))

    def test_pairwise_beliefs_value(self):
        beliefs = np.array([[0, 1],
                            [1, 0]])
        result = self.model.get_pairwise_beliefs(beliefs)
        np.testing.assert_array_equal(result, [[0, 0, 1, 0]])

    def test_pairwise_beliefs_value_2x2(self):
        self.model = LinearMRF(2, 2)
        beliefs = np.array([[0, 1],
                            [0, 1],
                            [1, 0],
                            [0, 1]])
        result = self.model.get_pairwise_beliefs(beliefs)
        np.testing.assert_array_equal(result, [[0, 0, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1],
                                               [0, 1, 0, 0]])


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = LinearMRF(1, 2)
    
    def test_inf(self):
        unary_beliefs = np.array([[0, 1],
                                  [0, 1]])
        unary_potentials = np.array([[1, 0],
                                     [1, 0]])
        pairwise_potentials = np.array([[2, 1, 1, 0]])
        correct = np.array([[1, 0],
                            [1, 0]])
        result = self.model.inference_itr(unary_beliefs, unary_potentials,
                                          pairwise_potentials)
        np.testing.assert_array_equal(correct, result)

    def test_local_score_1x2_1(self):
        unary_beliefs = np.array([[1, 0],
                                  [0, 1]])
        unary_potentials = np.array([[1, 0],
                                     [1, 0]])
        pairwise_potentials = np.array([[2, 1, 1, 0]])
        result = self.model.calculate_local_score(1, 0, unary_beliefs,
                                                  unary_potentials,
                                                  pairwise_potentials)
        self.assertEqual(3, result)

        result = self.model.calculate_local_score(1, 1, unary_beliefs,
                                                  unary_potentials,
                                                  pairwise_potentials)
        self.assertEqual(1, result)

    def test_local_score_1x2_2(self):
        unary_beliefs = np.array([[0, 1],
                                  [0, 1]])
        unary_potentials = np.array([[1, 0],
                                     [1, 0]])
        pairwise_potentials = np.array([[2, 1, 1, 0.5]])
        result = self.model.calculate_local_score(0, 0, unary_beliefs,
                                                  unary_potentials,
                                                  pairwise_potentials)
        self.assertEqual(1 + 1, result)

        result = self.model.calculate_local_score(0, 1, unary_beliefs,
                                                  unary_potentials,
                                                  pairwise_potentials)
        self.assertEqual(0.5, result)

    def test_local_score(self):
        self.model = LinearMRF(2, 2)
        node = 1
        assignment = 1
        unary_beliefs = np.array([[0, 1],
                                  [0, 1],
                                  [1, 0],
                                  [0, 1]])
        unary_potentials = np.array([[0.1, 0.8],
                                     [0.5, 0.71],
                                     [0.6, 0.2],
                                     [1, 0]])
        pairwise_potentials = np.array([[1, 1, 1, 0.2],
                                        [0, 0, 0, 0],
                                        [1, 1, 0.3, 0.5],
                                        [1, 1, 1.2, 1]])
        result = self.model.calculate_local_score(node, assignment,
                                                  unary_beliefs,
                                                  unary_potentials,
                                                  pairwise_potentials)
        self.assertEqual(result, 0.71 + 0.2 + 0.5)

        pairwise_potentials = np.array([[1, 1, 1, 0.7],
                                        [0, 0, 0, 0],
                                        [1, 1, 1, 7],
                                        [1, 1, 1, 2]])
        result = self.model.calculate_local_score(node, assignment,
                                                  unary_beliefs,
                                                  unary_potentials,
                                                  pairwise_potentials)
        self.assertEqual(result, 0.71 + 0.7 + 7)

        pairwise_potentials = np.array([[1, 1, 1, 5],
                                        [2, 2, 2, 2],
                                        [1, 1, 1, 7],
                                        [1, 1, 1, -1]])
        result = self.model.calculate_local_score(node, assignment,
                                                  unary_beliefs,
                                                  unary_potentials,
                                                  pairwise_potentials)
        self.assertEqual(result, 0.71 + 5 + 7)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.model = LinearMRF(1, 2)
 
    def test_learning_obj(self):
        img_features = np.array([[1, 0],
                                 [1, 0]])
        unary_beliefs = [tf.constant([[1, 0],
                                      [1, 0]])]
        pair_beliefs = [tf.constant([[1, 0, 0, 0]])]
        unary_potentials = [tf.constant([[1, 1],
                                         [1, 1]])]
        pairwise_potentials = tf.constant([[1, 1, 1, 1]])
        correct = 0
        result = self.model.build_training_obj(img_features, unary_beliefs,
                                               pair_beliefs, unary_potentials,
                                               pairwise_potentials)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result_val = sess.run(result)
        self.assertEqual(correct, result_val)

if __name__ == '__main__':
    unittest.main()
