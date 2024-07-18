"""bluepyefe.nwbreader tests"""
import unittest
import h5py
from pathlib import Path
from bluepyefe.reader import csv_lccr_reader


class TestCSVLCCRReaders(unittest.TestCase):
    def setUp(self):
        self.test_data = {
            'filepath': './tests/exp_data/csv_lccr/dummy/dummy_ch1_cols.txt',
            'protocol_name': 'Step',
            "dt": 0.1,
            "amplitudes": [10, -10, 20, -20, 30, -30, 40, -40, 50, -50, 60, -60, 70, -70, 80, -80, 90, -90, 100, -100, 150, 200, 250, 300, 400, 500, 600],
            "ljp": 14.0,
            "v_file": "dummy",
            "i_unit": "pA",
            "v_unit": "mV",
            "t_unit": "ms",
            "ton": 200,
            "toff": 800,
            "hypamp": -20,
            "remove_last_100ms": True,
        }

    def test_csv_lccr_reader(self):
        filepath = Path(self.test_data['filepath'])
        self.assertTrue(filepath.is_file(), f"{filepath} is not a valid file")

        result = csv_lccr_reader(self.test_data)
        self.assertIsInstance(result, list, f"Result for {filepath} should be a list")
        self.assertEqual(len(result), 27, f"Result for {filepath} should have 27 entries")

        for entry in result:
            self.assertIn('filename', entry)
            self.assertIn('voltage', entry)
            self.assertIn('current', entry)
            self.assertIn('t', entry)
            self.assertIn('dt', entry)
            self.assertIn('ton', entry)
            self.assertIn('toff', entry)
            self.assertIn('amp', entry)
            self.assertIn('hypamp', entry)
            self.assertIn('ljp', entry)
            self.assertIn('i_unit', entry)
            self.assertIn('v_unit', entry)
            self.assertIn('t_unit', entry)

    def test_csv_lccr_reader_empty_amplitudes(self):
        test_data = self.test_data.copy()
        test_data['amplitudes'] = []
        result = csv_lccr_reader(test_data)
        self.assertEqual(len(result), 0, "Result should be an empty list when amplitudes are empty")

    def test_csv_lccr_reader_file_not_found(self):
        test_data = self.test_data.copy()
        test_data['filepath'] = './non_existent_file.txt'
        with self.assertRaises(FileNotFoundError):
            csv_lccr_reader(test_data)

    def test_csv_lccr_reader_remove_last_100ms(self):
        test_data = self.test_data.copy()
        test_data['remove_last_100ms'] = True

        result = csv_lccr_reader(test_data)

        original_length = 14000
        expected_length = original_length - int(100 / test_data['dt'])

        for entry in result:
            self.assertEqual(len(entry['t']), expected_length, "Time array length should be reduced by 100 ms")
            self.assertEqual(len(entry['voltage']), expected_length, "Voltage array length should be reduced by 100 ms")
            self.assertEqual(len(entry['current']), expected_length, "Current array length should be reduced by 100 ms")


if __name__ == '__main__':
    unittest.main()