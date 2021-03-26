"""bluepyefe.reader tests"""

# import unittest
#
# from bluepyefe.reader import nwb_reader_BBP
#
#
# class CellTest(unittest.TestCase):
#
#     in_data = {
#         "filepath": "./tests/C311001B2-MT-C1.nwb",
#         "i_unit": "pA",
#         "t_unit": "s",
#         "v_unit": "mV",
#         "protocol_name": "IV"
#     }
#
#     def test_nwb_reader_BBP(self):
#
#         data = nwb_reader_BBP(self.in_data)
#
#         self.assertEqual(len(data), 48)
#         self.assertEqual(data[0]['dt'], 0.00025)
#
#     def test_nwb_reader_BBP_exception(self):
#
#         in_data_empty = {}
#
#         self.assertRaises(KeyError, nwb_reader_BBP, in_data_empty)
#
#
# if __name__ == "__main__":
#
#     unittest.main()
