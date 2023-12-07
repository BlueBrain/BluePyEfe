"""bluepyefe.extractor tests"""

import unittest
import glob
import json

import bluepyefe.extract
import bluepyefe.tools
from tests.utils import download_sahp_datafiles


def get_config(absolute_amplitude=False):

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

    if absolute_amplitude:
        targets = {
            "IDRest": {
                "amplitudes": [0.15, 0.25],
                "tolerances": [0.05],
                "efeatures": interesting_efeatures,
                "location": "soma",
            }
        }

    else:
        targets = {
            "IDRest": {
                "amplitudes": [150, 200, 250],
                "tolerances": [20.0],
                "efeatures": interesting_efeatures,
                "location": "soma",
            }
        }

    return files_metadata, bluepyefe.extract.convert_legacy_targets(targets)

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
                "t_unit": "s",
                "dt": 0.00025,
                "ljp": 14,
                "ton": 10,
                "tmid": 260,
                "tmid2": 360,
                "toff": 1360,
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
                "t_unit": "s",
                "dt": 0.00025,
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
                "tolerances": [0.01],
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


class ExtractorTest(unittest.TestCase):
    def test_extract(self):

        files_metadata, targets = get_config()

        cells = bluepyefe.extract.read_recordings(files_metadata=files_metadata)

        cells = bluepyefe.extract.extract_efeatures_at_targets(
            cells=cells, targets=targets
        )

        bluepyefe.extract.compute_rheobase(cells, protocols_rheobase=["IDRest"])

        self.assertEqual(len(cells), 2)
        self.assertEqual(len(cells[0].recordings), 5)
        self.assertEqual(len(cells[1].recordings), 5)

        self.assertLess(abs(cells[0].rheobase - 0.119), 0.01)
        self.assertLess(abs(cells[1].rheobase - 0.0923), 0.01)

        protocols = bluepyefe.extract.group_efeatures(
            cells,
            targets,
            use_global_rheobase=True,
            protocol_mode="mean"
        )

        _ = bluepyefe.extract.create_feature_protocol_files(
            cells=cells, protocols=protocols, output_directory="MouseCells"
        )

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

    def test_extract_auto_fail_rheobase(self):

        files_metadata, _ = get_config()

        efeatures, protocol_definitions, current = bluepyefe.extract.extract_efeatures(
            output_directory="./",
            files_metadata=files_metadata,
            rheobase_strategy="flush",
            rheobase_settings={"upper_bound_spikecount": 4}
        )

        self.assertEqual(len(efeatures), 0)

    def test_extract_auto(self):

        files_metadata, _ = get_config()

        auto_targets = bluepyefe.auto_targets.default_auto_targets()

        cells = bluepyefe.extract.read_recordings(
            files_metadata,
            recording_reader=None,
            map_function=map,
            efel_settings=bluepyefe.tools.DEFAULT_EFEL_SETTINGS
        )

        for cell in cells:
            cell.rheobase = 0.07
            cell.compute_relative_amp()

        recordings = []
        for c in cells:
            recordings += c.recordings

        for i in range(len(auto_targets)):
            auto_targets[i].select_ecode_and_amplitude(recordings)

        # Extract the efeatures and group them around the preset of targets.
        targets = []
        for at in auto_targets:
            targets += at.generate_targets()

        self.assertEqual(len(targets), 48)

    def test_extract_absolute(self):

        files_metadata, targets = get_config(True)

        cells = bluepyefe.extract.read_recordings(files_metadata=files_metadata)

        cells = bluepyefe.extract.extract_efeatures_at_targets(
            cells=cells, targets=targets
        )

        self.assertEqual(len(cells), 2)
        self.assertEqual(len(cells[0].recordings), 5)
        self.assertEqual(len(cells[1].recordings), 5)

        self.assertEqual(cells[0].rheobase, None)
        self.assertEqual(cells[1].rheobase, None)

        protocols = bluepyefe.extract.group_efeatures(
            cells,
            targets,
            absolute_amplitude=True,
            protocol_mode="mean"
        )

        _ = bluepyefe.extract.create_feature_protocol_files(
            cells=cells, protocols=protocols, output_directory="MouseCells"
        )

        for cell in cells:
            for r in cell.recordings:
                print(r.amp, r.efeatures)

        for protocol in protocols:
            if protocol.name == "IDRest" and protocol.amplitude == 0.25:
                for target in protocol.feature_targets:
                    if target.efel_feature_name == "Spikecount":
                        self.assertEqual(target.mean, 76.5)
                        self.assertAlmostEqual(target.std, 5.590169, 4)
                        break

        bluepyefe.extract.plot_all_recordings_efeatures(
            cells, protocols, output_dir="MouseCells/"
        )

        with open("MouseCells/features.json") as fp:
            features = json.load(fp)
        with open("MouseCells/protocols.json") as fp:
            protocols = json.load(fp)

        self.assertEqual(len(features), len(protocols))
    
    def test_extract_sahp(self):

        files_metadata, targets = get_sahp_config()

        cells = bluepyefe.extract.read_recordings(files_metadata=files_metadata)

        cells = bluepyefe.extract.extract_efeatures_at_targets(
            cells=cells, targets=targets
        )

        bluepyefe.extract.compute_rheobase(cells, protocols_rheobase=["IDthresh"])

        self.assertEqual(len(cells), 1)
        self.assertEqual(len(cells[0].recordings), 24)
        self.assertLess(abs(cells[0].rheobase - 0.1103), 0.01)
        self.assertLess(abs(cells[0].recordings[0].amp - 0.0953), 0.01)
        self.assertLess(abs(cells[0].recordings[0].amp2 - 0.3153), 0.01)
        self.assertLess(abs(cells[0].recordings[0].amp_rel - 86.4), 0.1)
        self.assertLess(abs(cells[0].recordings[0].amp2_rel - 285.8), 0.1)


        protocols = bluepyefe.extract.group_efeatures(
            cells,
            targets,
            use_global_rheobase=True,
            protocol_mode="mean"
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
