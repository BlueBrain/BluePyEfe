"""bluepyefe.extractor tests"""

import unittest
import glob
import json

import bluepyefe.extract
import bluepyefe.tools


def get_config():

    interesting_efeatures = {
        "Spikecount": {},
        "mean_frequency": {},
        "ISI_CV": {},
        "AP1_amp": {},
        "AP_width": {},
    }

    files_metadata1 = []
    for file in glob.glob("./tests/exp_data/B6/B6_Ch0_IDRest_*.ibw"):
        files_metadata1.append(
            {
                "i_file": file,
                "v_file": file.replace("Ch0", "Ch3"),
                "i_unit": "pA",
                "v_unit": "mV",
                "t_unit": "s",
                "dt": 0.00025,
                "ljp": 14.0,
            }
        )

    # Do the same for the second cell
    files_metadata2 = []
    for file in glob.glob("./tests/exp_data/B8/B8_Ch0_IDRest_*.ibw"):
        files_metadata2.append(
            {
                "i_file": file,
                "v_file": file.replace("Ch0", "Ch3"),
                "i_unit": "pA",
                "v_unit": "mV",
                "t_unit": "s",
                "dt": 0.00025,
                "ljp": 14.0,
            }
        )

    files_metadata = {
        "MouseNeuron1": {"IDRest": files_metadata1},
        "MouseNeuron2": {"IDRest": files_metadata2},
    }

    targets = {
        "IDRest": {
            "amplitudes": [150, 200, 250],
            "tolerances": [20.0],
            "efeatures": interesting_efeatures,
            "location": "soma",
        }
    }

    return files_metadata, bluepyefe.extract.convert_legacy_targets(targets)


class ExtractorTest(unittest.TestCase):
    def test_extract(self):

        files_metadata, targets = get_config()

        cells = bluepyefe.extract.read_recordings(files_metadata=files_metadata)

        cells = bluepyefe.extract.extract_efeatures_at_targets(
            cells=cells, targets=targets
        )

        bluepyefe.extract.compute_rheobase(cells, protocols_rheobase=["IDRest"])

        protocols = bluepyefe.extract.group_efeatures(
            cells,
            targets,
            use_global_rheobase=True,
            protocol_mode="mean"
        )

        _ = bluepyefe.extract.create_feature_protocol_files(
            cells=cells, protocols=protocols, output_directory="MouseCells"
        )

        self.assertEqual(len(cells), 2)
        self.assertEqual(len(cells[0].recordings), 5)
        self.assertEqual(len(cells[1].recordings), 5)

        self.assertLess(abs(cells[0].rheobase - 0.119), 0.01)
        self.assertLess(abs(cells[1].rheobase - 0.0923), 0.01)

        for protocol in protocols:
            if protocol.name == "IDRest" and protocol.amplitude == 250.:
                for target in protocol.feature_targets:
                    if target.efel_feature_name == "Spikecount":
                        self.assertEqual(target.mean, 78.5)
                        self.assertEqual(target.std, 3.5)
                        break

        bluepyefe.extract.plot_all_recordings_efeatures(
            cells, protocols, output_dir="MouseCells/"
        )

        with open("MouseCells/features.json") as fp:
            features = json.load(fp)
        with open("MouseCells/protocols.json") as fp:
            protocols = json.load(fp)

        self.assertEqual(len(features), len(protocols))


if __name__ == "__main__":
    unittest.main()
