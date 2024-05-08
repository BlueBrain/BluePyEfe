"""bluepyefe.cell tests"""

import unittest
import pytest

import bluepyefe.cell
import bluepyefe.recording


class EfelSettingTest(unittest.TestCase):
    def setUp(self):

        self.cell = bluepyefe.cell.Cell(name="MouseNeuron")

        file_metadata = {
            "i_file": "./tests/exp_data/B95_Ch0_IDRest_107.ibw",
            "v_file": "./tests/exp_data/B95_Ch3_IDRest_107.ibw",
            "i_unit": "pA",
            "v_unit": "mV",
            "t_unit": "s",
            "dt": 0.00025,
            "ljp": 14.0,
        }

        self.cell.read_recordings(
            protocol_data=[file_metadata],
            protocol_name="IDRest"
        )

    def test_efel_threshold(self):

        self.cell.recordings[0].efeatures = {}

        self.cell.extract_efeatures(
            protocol_name="IDRest",
            efeatures=["Spikecount", "AP1_amp"],
            efel_settings={'Threshold': 40.}
        )

        recording = self.cell.recordings[0]
        self.assertEqual(recording.efeatures["Spikecount"], 0.)
        self.assertLess(abs(recording.efeatures["AP1_amp"] - 66.68), 0.01)

    def test_efel_strictstim(self):

        self.cell.recordings[0].efeatures = {}

        self.cell.extract_efeatures(
            protocol_name="IDRest",
            efeatures=["Spikecount"],
            efel_settings={
                'stim_start': 0,
                'stim_end': 100,
                'strict_stiminterval': True
            }
        )

        self.assertEqual(self.cell.recordings[0].efeatures["Spikecount"], 0.)

    def test_efel_threshold(self):

        self.cell.recordings[0].efeatures = {}

        self.cell.extract_efeatures(
            protocol_name="IDRest",
            efeatures=["Spikecount"],
            efel_settings={'Threshold': 40}
        )

        recording = self.cell.recordings[0]
        self.assertEqual(recording.efeatures["Spikecount"], 0.)

    def test_efel_incorrect_threshold(self):

        self.cell.recordings[0].efeatures = {}

        with pytest.raises(ValueError):
            self.cell.extract_efeatures(
                protocol_name="IDRest",
                efeatures=["Spikecount"],
                efel_settings={'Threshold': ["40."]}
            )

if __name__ == "__main__":
    unittest.main()
