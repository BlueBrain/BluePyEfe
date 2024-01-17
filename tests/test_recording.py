"""bluepyefe.cell tests"""

import unittest

from numpy.testing import assert_array_almost_equal
from pytest import approx

import bluepyefe.cell
import bluepyefe.recording
from bluepyefe.reader import igor_reader
from bluepyefe.ecode.step import Step


class RecordingTest(unittest.TestCase):
    def setUp(self):

        config_data = {
            "i_file": "./tests/exp_data/B95_Ch0_IDRest_107.ibw",
            "v_file": "./tests/exp_data/B95_Ch3_IDRest_107.ibw",
            "i_unit": "pA",
            "v_unit": "mV",
            "t_unit": "s",
            "dt": 0.00025,
            "ljp": 14.0,
        }

        self.recording = Step(
            config_data,
            igor_reader(config_data)[0],
            protocol_name="Step",
            efel_settings={}
        )

    def test_step_ecode(self):
        self.assertTrue(isinstance(self.recording, bluepyefe.recording.Recording))
        self.assertEqual(len(self.recording.voltage), len(self.recording.current))
        self.assertEqual(len(self.recording.voltage), len(self.recording.t))
        self.assertLess(abs(self.recording.ton - 700.0), 10.0)
        self.assertLess(abs(self.recording.toff - 2700.0), 10.0)
        self.assertLess(abs(self.recording.hypamp + 0.03), 0.005)
        self.assertLess(abs(self.recording.amp - 0.033), 0.005)

    def test_get_params(self):
        params = self.recording.get_params()
        self.assertEqual(len(params), len(self.recording.export_attr))

    def test_generate(self):
        t, c = self.recording.generate()
        self.assertEqual(len(t), len(c))
        self.assertEqual(max(c), self.recording.amp + self.recording.hypamp)

    def test_in_target(self):
        self.recording.amp_rel = 100.
        self.assertTrue(self.recording.in_target(101, 2))
        self.assertFalse(self.recording.in_target(-100, 50))
        self.assertFalse(self.recording.in_target(90, 2))


class RecordingTestNWB(unittest.TestCase):

    def setUp(self):
        cell = bluepyefe.cell.Cell(name="MouseNeuron")
        file_metadata = {
                        "filepath": "./tests/exp_data/hippocampus-portal/99111002.nwb",
                        "i_unit": "A",
                        "v_unit": "V",
                        "t_unit": "s",
                        "ljp": 0.0,
                        "protocol_name": "Step",
                    }
        cell.read_recordings(protocol_data=[file_metadata], protocol_name="Step")
        self.cell = cell

    def test_set_autothreshold(self):
        """Test the auto_threshold detection in Recording."""
        assert self.cell.recordings[0].auto_threshold == approx(4.999999)
        assert self.cell.recordings[15].auto_threshold == approx(26.5)

    def test_compute_spikecount(self):
        """Test Recording.compute_spikecount()."""
        assert self.cell.recordings[1].spikecount == 2
        assert_array_almost_equal(self.cell.recordings[1].peak_time, [85.4, 346.1])


if __name__ == "__main__":
    unittest.main()
