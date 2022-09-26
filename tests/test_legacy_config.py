"""bluepyefe.protocol tests"""

import unittest
import glob

from bluepyefe.translate_legacy_config import translate_legacy_config
import bluepyefe.extract
import bluepyefe.tools


class TestTranslateLegacyConfig(unittest.TestCase):

    def setUp(self):

        files_metadata = {}
        for cell_name in ["B6", "B8"]:
            for path in glob.glob(f"./tests/exp_data/{cell_name}/{cell_name}_Ch0_IDRest_*.ibw"):

                if cell_name not in files_metadata:
                    files_metadata[cell_name] = {"experiments": {"IDRest": {"files": []}}}

                files_metadata[cell_name]["experiments"]["IDRest"]["files"].append(
                    {
                        "i_file": path,
                        "v_file": path.replace("Ch0", "Ch3"),
                        "i_unit": "pA",
                        "v_unit": "mV",
                        "t_unit": "s",
                        "dt": 0.00025,
                        "ljp": 14.0,
                    }
                )

        self.config = {
            "cells": files_metadata,
            "features": {"IDRest": ["AP_amplitude", "Spikecount"]},
            "options": {
                "target": [150, 200, 250],
                "tolerance": [20, 20, 20],
                "onoff": {"IDRest": [700, 2700]},
                "expthreshold": ["IDRest"]
            },
            "path": "./",
        }

    def test_translate(self):
        translated_config = translate_legacy_config(self.config)

        cells = bluepyefe.extract.read_recordings(
            files_metadata=translated_config["files_metadata"])

        cells = bluepyefe.extract.extract_efeatures_at_targets(
            cells=cells, targets=translated_config["targets"]
        )

        bluepyefe.extract.compute_rheobase(cells, protocols_rheobase=["IDRest"])

        self.assertEqual(len(cells), 2)
        self.assertEqual(len(cells[0].recordings), 5)
        self.assertEqual(len(cells[1].recordings), 5)

        self.assertLess(abs(cells[0].rheobase - 0.119), 0.01)
        self.assertLess(abs(cells[1].rheobase - 0.0923), 0.01)


if __name__ == "__main__":
    unittest.main()
