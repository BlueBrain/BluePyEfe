"""bluepyefe.protocol tests"""

import unittest

from bluepyefe.ecode.step import Step
from bluepyefe.protocol import Protocol
from bluepyefe.target import EFeatureTarget


class TestProtocol(unittest.TestCase):

    def setUp(self):
        target = EFeatureTarget(
            efeature_name='test_spikecount',
            efel_feature_name='Spikecount',
            protocol_name='IDRest',
            amplitude=150.,
            tolerance=10.
        )

        self.protocol = Protocol(
            name='IDRest',
            feature_targets=[target],
            amplitude=150.,
            tolerance=10.,
            mode="mean"
        )

    def test_append_clear(self):
        rec = Step(config_data={}, reader_data={})
        rec.efeatures = {"test_spikecount": 10.}

        self.protocol.append(rec)
        self.protocol.append(rec)
        self.assertEqual(self.protocol.n_match, 2)

    def test_str(self):
        print(self.protocol)


if __name__ == "__main__":
    unittest.main()
