"""Simple unit tests for students."""

import unittest
import numpy as np
import train_eval_model
from utils import data_tools, io_tools
from models import linear_regression


class IoToolsTests(unittest.TestCase):
    def setUp(self):
        self.dataset = io_tools.read_dataset("data/train.csv")

    def test_read_dataset_not_none(self):
        self.assertIsNotNone(self.dataset)

    def test_first_row(self):
        keys = sorted(list(self.dataset.keys()))
        val0 = (self.dataset[keys[0]])
        val0_true = ('1', '1Fam', '7', '1710', '548', '208500')
        self.assertEqual(val0, val0_true)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = linear_regression.LinearRegression(5, 'zeros')

    def test_forward_shape(self):
        x = np.zeros((10, 5))
        y_hat = self.model.forward(x)
        self.assertEqual(y_hat.shape, (10, 1))

    def test_forward_zero(self):
        x = np.zeros((10, 5))
        y = np.zeros((10, 1))
        y_hat = self.model.forward(x)
        np.testing.assert_array_equal(y, y_hat)

    def test_uniform_init(self):
        self.model = linear_regression.LinearRegression(5, 'uniform')
        np.testing.assert_equal(np.any(np.equal(np.zeros((6, 1)), self.model.w)), False)

    def test_init_ones(self):
        self.model = linear_regression.LinearRegression(5, 'ones')
        np.testing.assert_array_equal(np.ones((6, 1)), self.model.w)


class DataToolsTests(unittest.TestCase):
    def setUp(self):
        self.dataset = io_tools.read_dataset("data/train.csv")
        self.N = len(self.dataset)

    def test_one_hot_bldg_type(self):
        val = data_tools.one_hot_bldg_type('Duplx')
        np.testing.assert_array_equal(val, [0, 0, 1, 0, 0])

        val = data_tools.one_hot_bldg_type('TwnhsI')
        np.testing.assert_array_equal(val, [0, 0, 0, 0, 1])

    def test_preprocess_dataset_shape(self):
        feature_columns = ['Id', 'GarageArea']
        data = data_tools.preprocess_data(self.dataset,
                                          feature_columns=feature_columns)
        self.assertEqual(len(data), 2)
        # check x
        self.assertEqual(data[0].shape, (self.N, 2))
        # check y
        self.assertEqual(data[1].shape, (self.N, 1))

    def test_preprocess_dataset_x_not_price(self):
        feature_columns = ['Id', 'GarageArea', 'SalePrice']
        data = data_tools.preprocess_data(self.dataset,
                                          feature_columns=feature_columns)
        self.assertEqual(data[0].shape, (self.N, 2))

    def test_preprocess_dataset_one_hot_encoding(self):
        feature_columns = ['BldgType']
        data = data_tools.preprocess_data(self.dataset,
                                          feature_columns=feature_columns)
        self.assertEqual(data[0].shape, (self.N, 5))

        feature_columns = ['BldgType', 'Id']
        data = data_tools.preprocess_data(self.dataset,
                                          feature_columns=feature_columns)
        self.assertEqual(data[0].shape, (self.N, 6))

    def test_preprocess_dataset_squared(self):
        feature_columns = ['OverallQual']
        data = data_tools.preprocess_data(self.dataset,
                                          feature_columns=feature_columns,
                                          squared_features=True)
        keys = sorted(list(self.dataset.keys()))
        val0 = float(self.dataset[keys[0]][2]) ** 2
        self.assertEqual(49, val0)

    def test_preprocess_dataset_y_number(self):
        feature_columns = ['Id', 'GarageArea', 'SalePrice']
        data = data_tools.preprocess_data(self.dataset,
                                          feature_columns=feature_columns)
        self.assertEqual(type(data[1][0][0]), np.float32)


class LinearModelTests(unittest.TestCase):
    def setUp(self):
        cols = ['GarageArea', 'OverallQual', 'BldgType']
        self.dataset = io_tools.read_dataset("data/train.csv")
        self.processed_data = data_tools.preprocess_data(self.dataset, feature_columns=cols)
        self.N = self.processed_data[0].shape[0]
        self.ndims = self.processed_data[0].shape[1]
        self.model = linear_regression.LinearRegression(self.ndims, "zeros")

    def test_train_model_analytic_w_shape(self):
        train_eval_model.train_model_analytic(self.processed_data, self.model)
        self.assertEqual(self.model.w.shape, (8, 1))

    def test_forward_shape(self):
        f = self.model.forward(self.processed_data[0])
        self.assertEqual(f.shape, (self.N, 1))

    def test_total_loss(self):
        self.model = linear_regression.LinearRegression(2, 'zeros', w_decay_factor=0.1)
        self.model.w = np.array([[2], [-2]])
        y = np.array([[1], [1], [1], [-1], [-1], [-1]])
        f = np.array([[0.5], [0], [0.5], [-1], [-1], [-1]])
        loss = self.model.total_loss(f, y)
        self.assertEqual(round(loss, 5), 0.5 * (0.5**2. + 1 + 0.5**2. + 0 + 0 + 8 * 0.1))


if __name__ == '__main__':
    unittest.main()
