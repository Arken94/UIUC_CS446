"""Simple unit tests for students. (This file will not be graded.)"""

import unittest
import numpy as np
from utils import io_tools
from utils import data_tools
from models import support_vector_machine
from train_eval_model import qp_helper


class IoToolsTests(unittest.TestCase):
    def setUp(self):
        self.dataset = io_tools.read_dataset(
            "data/train.txt", "data/image_data/")

    def test_read_dataset_not_none(self):
        self.assertIsNotNone(self.dataset)

    def test_data_shape(self):
        image = self.dataset['image']
        label = self.dataset['label']
        self.assertEqual(image.shape[0], label.shape[0])

    def test_data_image_shape(self):
        images = self.dataset['image']
        for img in images:
            self.assertEqual(img.shape, (8, 8, 3))


class DataToolsTests(unittest.TestCase):
    def setUp(self):
        self.dataset = io_tools.read_dataset(
            "data/train.txt", "data/image_data/")

    def test_raw_shape(self):
        original_shape = self.dataset['image'].shape
        data = data_tools.preprocess_data(self.dataset, process_method='raw')[
            'image']
        self.assertEqual(len(data.shape), 2)
        self.assertEqual(data.shape[0], original_shape[0])
        self.assertEqual(data.shape[1],
                         original_shape[1] * original_shape[2] * original_shape[
                             3])

    def test_default_shape(self):
        original_shape = self.dataset['image'].shape
        data = data_tools.preprocess_data(self.dataset,
                                          process_method='default')['image']
        self.assertEqual(len(data.shape), 2)
        self.assertEqual(data.shape[0], original_shape[0])
        self.assertEqual(data.shape[1],
                         original_shape[1] * original_shape[2] * original_shape[
                             3])

    def test_compute_image_mean_shape(self):
        mean = data_tools.compute_image_mean(self.dataset)
        self.assertEqual(mean.shape, self.dataset['image'][0].shape)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = support_vector_machine.SupportVectorMachine(5, 'zeros')

    def test_forward_shape(self):
        x = np.zeros((10, 5))
        y_hat = self.model.forward(x)
        self.assertEqual(y_hat.shape, (10, 1))

    def test_forward_zero(self):
        x = np.zeros((10, 5))
        y = np.zeros((10, 1))
        y_hat = self.model.forward(x)
        np.testing.assert_array_equal(y, y_hat)

    def test_predict(self):
        f = np.array([0.4, -100, np.inf, 0, 1, -np.inf]).reshape(6, 1)
        y_pred = self.model.predict(f)
        np.testing.assert_array_equal(y_pred, np.array(
            [[1], [-1], [1], [1], [1], [-1]]))


class QpTests(unittest.TestCase):
    def setUp(self):
        self.dataset = io_tools.read_dataset(
            "data/train.txt", "data/image_data/")
        self.dataset = data_tools.preprocess_data(self.dataset, 'raw')
        self.model = support_vector_machine.SupportVectorMachine(
            8 * 8 * 3, 'zeros')

    def test_qp(self):
        P, q, G, h = qp_helper(self.dataset, self.model)
        self.assertEqual(P.shape[0], q.shape[0])
        self.assertEqual(G.shape[0], h.shape[0])

    def test_qp_h_values(self):
        _, _, _, h = qp_helper(self.dataset, self.model)
        N = self.dataset['label'].shape[0]
        np.testing.assert_array_equal(h[:N], -np.ones((N, 1)))
        np.testing.assert_array_equal(h[N:], np.zeros((N, 1)))


if __name__ == '__main__':
    unittest.main()
