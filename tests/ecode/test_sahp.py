"""bluepyefe.ecode.sAHP tests"""

import unittest
import glob
import json

import bluepyefe.extract
import bluepyefe.tools
from tests.utils import download_sahp_datafiles


def get_sahp_config(absolute_amplitude=False):
    download_sahp_datafiles()

    interesting_efeatures = {
        "Spikecount": {},
        "mean_frequency": {},
        "ISI_CV": {},
        "AP1_amp": {},
        "AP_width": {},
    }

    files_metadata1 = []
    for file in glob.glob("./tests/exp_data/X/X_sAHP_ch0_*.ibw"):
        files_metadata1.append(
            {
                "i_file": file,
                "v_file": file.replace("ch0", "ch1"),
                "i_unit": "A",
                "v_unit": "V",
                "t_unit": "ms",
                "dt": 0.25,
                "ljp": 14,
                "ton": 10, # in ms
                "tmid": 260, # in ms
                "tmid2": 360, # in ms
                "toff": 1360, # in ms
            }
        )
    files_metadata2 = []
    for file in glob.glob("./tests/exp_data/X/X_IDthresh_ch0_*.ibw"):
        files_metadata2.append(
            {
                "i_file": file,
                "v_file": file.replace("ch0", "ch1"),
                "i_unit": "A",
                "v_unit": "V",
                "t_unit": "ms",
                "dt": 0.25,
                "ljp": 14,
            }
        )

    files_metadata = {
        "MouseNeuron1": {"sAHP": files_metadata1, "IDthresh": files_metadata2},
    }

    if absolute_amplitude:
        targets = {
            "sAHP": {
                "amplitudes": [0.315, 0.225, 0.5, 0.69, 0.41, 0.595],
                "tolerances": [0.1],
                "efeatures": interesting_efeatures,
                "location": "soma",
            }
        }

    else:
        targets = {
            "sAHP": {
                "amplitudes": [285, 200, 450, 625, 370, 540],
                "tolerances": [10.0],
                "efeatures": interesting_efeatures,
                "location": "soma",
            }
        }

    return files_metadata, bluepyefe.extract.convert_legacy_targets(targets)


class SAHPTest(unittest.TestCase):
    def test_extract_sahp(self):
        for absolute_amplitude in [True, False]:
            with self.subTest(absolute_amplitude=absolute_amplitude):
                self.run_test_with_absolute_amplitude(absolute_amplitude)

    def run_test_with_absolute_amplitude(self, absolute_amplitude):
        files_metadata, targets = get_sahp_config(absolute_amplitude)

        cells = bluepyefe.extract.read_recordings(files_metadata=files_metadata)

        cells = bluepyefe.extract.extract_efeatures_at_targets(
            cells=cells, targets=targets
        )

        bluepyefe.extract.compute_rheobase(cells, protocols_rheobase=["IDthresh"])

        self.assertEqual(len(cells), 1)
        self.assertEqual(len(cells[0].recordings), 24)
        self.assertLess(abs(cells[0].rheobase - 0.1103), 0.01)

        # amplitude test for one recording
        # sort the recordings because they can be in any order,
        # and we want to select the same one each time we test
        sahp_recs = [rec for rec in cells[0].recordings if rec.protocol_name == "sAHP"]
        rec1 = sorted(sahp_recs, key=lambda x: x.amp2)[1]
        self.assertLess(abs(rec1.amp - 0.0953), 0.01)
        self.assertLess(abs(rec1.amp2 - 0.3153), 0.01)
        self.assertLess(abs(rec1.amp_rel - 86.4), 0.1)
        self.assertLess(abs(rec1.amp2_rel - 285.8), 0.1)


        protocols = bluepyefe.extract.group_efeatures(
            cells,
            targets,
            use_global_rheobase=True,
            protocol_mode="mean",
            absolute_amplitude=absolute_amplitude
        )

        _ = bluepyefe.extract.create_feature_protocol_files(
            cells=cells, protocols=protocols, output_directory="MouseCells_sAHP"
        )

        for protocol in protocols:
            if protocol.name == "sAHP" and protocol.amplitude == 625:
                for target in protocol.feature_targets:
                    if target.efel_feature_name == "Spikecount":
                        self.assertEqual(target.mean, 6)
                        break

        bluepyefe.extract.plot_all_recordings_efeatures(
            cells, protocols, output_dir="MouseCells_sAHP/"
        )

        with open("MouseCells_sAHP/features.json") as fp:
            features = json.load(fp)
        with open("MouseCells_sAHP/protocols.json") as fp:
            protocols = json.load(fp)

        self.assertEqual(len(features), len(protocols))

if __name__ == "__main__":
    unittest.main()