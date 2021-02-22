"""bluepyefe.cell tests"""

import bluepyefe.cell
import bluepyefe.recording
import unittest


class CellTest(unittest.TestCase):
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

        self.efeatures = {"Spikecount": None, "AP1_amp": None}
        self.cell.extract_efeatures(protocol_name="IDRest", efeatures=self.efeatures)

    def test_step_ecode(self):
        recording = self.cell.recordings[0]

        self.assertTrue(isinstance(recording, bluepyefe.recording.Recording))
        self.assertEqual(len(recording.voltage), len(recording.current))
        self.assertEqual(len(recording.voltage), len(recording.t))
        self.assertLess(abs(recording.ton - 700.0), 10.0)
        self.assertLess(abs(recording.toff - 2700.0), 10.0)
        self.assertLess(abs(recording.hypamp + 0.03), 0.005)
        self.assertLess(abs(recording.amp - 0.033), 0.005)

    def test_efeature_extraction(self):
        recording = self.cell.recordings[0]
        self.assertEqual(len(self.efeatures), len(recording.efeatures))
        self.assertEqual(recording.efeatures["Spikecount"], 9.0)
        self.assertLess(abs(recording.efeatures["AP1_amp"] - 66.4), 2.0)

    def test_amp_threshold(self):
        recording = self.cell.recordings[0]
        self.cell.compute_rheobase(["IDRest"])
        self.cell.compute_relative_amp()
        self.assertEqual(recording.amp, self.cell.rheobase)
        self.assertEqual(recording.amp_rel, 100.0)


if __name__ == "__main__":
    unittest.main()
