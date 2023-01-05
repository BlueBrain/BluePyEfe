"""bluepyefe.target tests"""

import unittest
import numpy
import math

from bluepyefe.target import EFeatureTarget


class TestEFeatureTarget(unittest.TestCase):

    def setUp(self):
        self.target = EFeatureTarget(
            efeature_name='test_spikecount',
            efel_feature_name='Spikecount',
            protocol_name='IDRest',
            amplitude=150.,
            tolerance=10.
        )

    def test_init(self):
        self.assertEqual(self.target.sample_size, 0)

    def test_nan(self):
        self.target.append(numpy.nan)
        self.target.append(math.nan)
        self.assertEqual(self.target.sample_size, 0)

    def test_append_clear(self):
        self.target.append(1.)
        self.target.append(2.)
        with self.assertRaises(TypeError) as context:
            self.target.append([1.])
        self.assertTrue('Expected value' in str(context.exception))
        self.assertEqual(self.target.sample_size, 2)
        self.target.clear()
        self.assertEqual(self.target.sample_size, 0)

    def test_mean_std(self):
        self.assertTrue(numpy.isnan(self.target.mean))
        self.assertTrue(numpy.isnan(self.target.std))
        self.target.append(1.)
        self.target.append(2.)
        self.assertEqual(self.target.mean, 1.5)
        self.assertEqual(self.target.std, 0.5)

    def test_dict(self):
        self.target.append(1.)
        self.target.append(2.)
        dict_form = self.target.as_dict()
        self.assertEqual(len(dict_form), 5)
        self.assertEqual(len(dict_form['val']), 2)
        self.assertEqual(len(dict_form['efel_settings']), 0)

    def test_str(self):
        print(self.target)
        self.target.append(1.)
        self.target.append(2.)
        print(self.target)


if __name__ == "__main__":
    unittest.main()
