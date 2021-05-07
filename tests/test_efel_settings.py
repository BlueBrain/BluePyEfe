"""bluepyefe.cell tests"""

import unittest

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

        self.cell.read_recordings(protocol_data=[file_metadata], protocol_name="IDRest")

    def test_efel_threshold(self):
        
        self.cell.recordings[0].efeatures = {}
        efeatures = {"Spikecount": {'Threshold': 40.}, "AP1_amp": {'Threshold': 0.}}
        self.cell.extract_efeatures(protocol_name="IDRest", efeatures=efeatures)

        recording = self.cell.recordings[0]
        self.assertEqual(recording.efeatures["Spikecount"], 0.)
        self.assertLess(abs(recording.efeatures["AP1_amp"] - 66.68), 0.01)

    def test_efel_strictstim(self):
        
        self.cell.recordings[0].efeatures = {}
        efeatures = {"Spikecount": {'stim_start': 0, 'stim_end': 500}}
        self.cell.extract_efeatures(protocol_name="IDRest", efeatures=efeatures)

        recording = self.cell.recordings[0]
        self.assertEqual(recording.efeatures["Spikecount"], 0.)

    def test_efel_global_setting(self):

        self.cell.recordings[0].efeatures = {}

        efeatures = {"Spikecount": {}}

        self.cell.extract_efeatures(
            protocol_name="IDRest",
            efeatures=efeatures,
            global_efel_settings={'Threshold': 40.}
        )

        recording = self.cell.recordings[0]
        self.assertEqual(recording.efeatures["Spikecount"], 0.)

if __name__ == "__main__":
    unittest.main()
