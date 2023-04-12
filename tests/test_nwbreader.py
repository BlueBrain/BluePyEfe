"""bluepyefe.nwbreader tests"""
import unittest
import h5py
from pathlib import Path
from bluepyefe.reader import nwb_reader


class TestNWBReaders(unittest.TestCase):
    def setUp(self):
        self.test_data = {
            'filepath': './tests/exp_data/hippocampus-portal/99111002.nwb',
            'protocol_name': 'Step',
        }

    def test_nwb_reader(self):
        filepath = Path(self.test_data['filepath'])
        self.assertTrue(filepath.is_file(), f"{filepath} is not a valid file")

        result = nwb_reader(self.test_data)

        self.assertIsInstance(result, list, f"Result for {filepath} should be a list")
        self.assertEqual(len(result), 16, f"Result for {filepath} should have 16 entries")

        for entry in result:
            self.assertIn('voltage', entry)
            self.assertIn('current', entry)
            self.assertIn('dt', entry)
            self.assertIn('id', entry)
            self.assertIn('i_unit', entry)
            self.assertIn('v_unit', entry)
            self.assertIn('t_unit', entry)


if __name__ == '__main__':
    unittest.main()