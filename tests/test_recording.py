"""bluepyefe.cell tests"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
