import unittest
from logistic_regression import LogisticRegression
import numpy as np


class TestingLogisticRegression(unittest.TestCase):

    def test_sigmoid(self):
        model = LogisticRegression()
        self.assertEqual(model.sigmoid(0), 0.5)
        self.assertEqual(model.sigmoid(10000), 1)
        self.assertEqual(model.sigmoid(-10000), 0)

        mat = np.matrix('-1000 0 1000')
        self.assertEqual(model.sigmoid(mat).all(), np.matrix('0 0.5 1').all())

