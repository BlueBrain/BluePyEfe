"""Efeature extraction functions"""

"""
Copyright (c) 2020, EPFL/Blue Brain Project

 This file is part of BluePyEfe <https://github.com/BlueBrain/BluePyEfe>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import functools
import logging
import pathlib
import gc

import numpy

from bluepyefe import tools
from bluepyefe.cell import Cell
from bluepyefe.plotting import plot_all_recordings
from bluepyefe.plotting import plot_all_recordings_efeatures
from bluepyefe.protocol import Protocol
from bluepyefe.target import EFeatureTarget

logger = logging.getLogger(__name__)


def _create_cell(cell_definition, recording_reader):
    """
    Initialize a Cell object and populate it with the content of the associated
    recording files.

    The present function exist to be use by the map_function.
    """

    cell_name = cell_definition[0]
    cell = cell_definition[1]

    out_cell = Cell(name=cell_name)

    for prot_name in cell.keys():
        out_cell.read_recordings(
            protocol_data=cell[prot_name],
            protocol_name=prot_name,
            recording_reader=recording_reader,
        )

    return out_cell


def _extract_efeatures_cell(cell, targets, efel_settings=None):
    """
    Compute the efeatures on all the recordings of a Cell.

    The present function exists to be use by the map_function.
    """

    if efel_settings is None:
        efel_settings = {}

    # Group targets per same protocol and same efel settings for efficiency
    setting_groups = []
    for target in targets:

        for i, group in enumerate(setting_groups):
            if target["efel_settings"] == group['efel_settings'] and \
                    target["protocol"] == group['protocol']:
                setting_groups[i]["efeatures"].append(target["efeature"])
                break
        else:
            setting_groups.append({
                'efel_settings': target["efel_settings"],
                'protocol': target["protocol"],
                'efeatures': [target["efeature"]]
            })

    for group in setting_groups:
        cell.extract_efeatures(
            group['protocol'],
            group["efeatures"],
            efel_settings={**efel_settings, **group["efel_settings"]}
        )

    return cell


def _saving_data(output_directory, feat, stim, currents):
    """
    Save the features, protocols and current to json files.
    """

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    features_path = output_directory / "features.json"
    tools.dict_to_json(feat, features_path)
    logger.info("Saved efeatures to {}".format(features_path))

    protocols_path = output_directory / "protocols.json"
    tools.dict_to_json(stim, protocols_path)
    logger.info("Saving protocols to {}".format(protocols_path))

    currents_path = output_directory / "holding_threshold_currents.json"
    tools.dict_to_json(currents, currents_path)
    logger.info(
        "Saving threshold and holding currents to {}" "".format(currents_path)
    )


def read_recordings(files_metadata, recording_reader=None, map_function=map):
    """
    Read recordings from a group of files. The files are expected to be
    identified by both a cell id and a protocol name (see files_metadata).

    Args:
        files_metadata (dict): define for which cell and protocol each file
            has to be used. Of the form:
            {
                cell_id: {
                    protocol_name: [{file_metadata1}, {file_metadata1}]
                }
            }
            A same file path might be present in the file metadata for
            different protocols.
            The entries required in the file_metadata are specific to each
            recording_reader (see bluepyemodel/reader.py to know which one are
            needed for your recording_reader).
        recording_reader (function): custom recording reader function.
            It's inner working has to match the metadata entered in
            files_metadata.
        map_function (function): Function used to map (parallelize) the
            recording reading operations. Note: the parallelization is
            done across cells an not across files.

    Return:
         cells (list): list of Cell objects containing the data of the
         recordings
    """

    cells = map_function(
        functools.partial(
            _create_cell,
            recording_reader=recording_reader,
        ),
        list(files_metadata.items()),
    )

    return list(cells)


def extract_efeatures_at_targets(
        cells,
        targets,
        map_function=map,
        efel_settings=None
):
    """
    Extract efeatures from recordings following the protocols, amplitudes and
    efeature names specified in the targets.

    Args:
        cells (list): list of Cells containing the recordings from which the
            efeatures will be extracted.
        targets (dict): define the efeatures to extract as well as which
            protocols and current amplitude they should be extracted for. Of
            the form:
            [{
                "efeature": "AP_amplitude",
                "protocol": "IDRest",
                "amplitude": 150.,
                "tolerance": 10.,
                "efel_settings": {
                    'stim_start': 200.,
                    'stim_end': 500.,
                    'Threshold': -10.
                }
            }]
        map_function (function): Function used to map (parallelize) the
            feature extraction operations. Note: the parallelization is
            done across cells an not across efeatures.
        efel_settings (dict): efel settings in the form
            {setting_name: setting_value}. If settings are also informed
            in the targets per efeature, the latter will have priority.
    """

    for target in targets:

        if 'location' not in target:
            target["location"] = "soma"

        if 'efel_settings' not in target:
            target['efel_settings'] = {}

    cells = map_function(
        functools.partial(
            _extract_efeatures_cell,
            targets=targets,
            efel_settings=efel_settings
        ),
        cells,
    )

    return list(cells)


def compute_rheobase(cells, protocols_rheobase, spike_threshold=1):
    """
    For each cell, finds the smallest current inducing a spike (rheobase).
    This currents are then use it to compute the relative amplitude of
    the stimuli.

    Args:
        cells (list): list of Cells containing for which the rheobase will be
            computed
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        spike_threshold (int): number of spikes above which a recording
            is considered to compute the rheobase.
    """

    for cell in cells:
        cell.compute_rheobase(protocols_rheobase, spike_threshold)
        cell.compute_relative_amp()


def _build_protocols(targets, global_rheobase, protocol_mode):
    """Build a list of Protocols that matches the expected targets"""

    protocols = []

    for target in targets:

        efeature_target = EFeatureTarget(
            efel_feature_name=target['efeature'],
            protocol_name=target['protocol'],
            amplitude=target['amplitude'],
            tolerance=target['tolerance']
        )

        for i, p in enumerate(protocols):
            if p.name == target['protocol'] and \
                    p.amplitude == target['amplitude']:
                protocols[i].feature_targets.append(efeature_target)
                break
        else:
            protocols.append(
                Protocol(
                    name=target['protocol'],
                    amplitude=target['amplitude'],
                    tolerance=target['tolerance'],
                    feature_targets=[efeature_target],
                    global_rheobase=global_rheobase,
                    mode=protocol_mode
                )
            )

    return protocols


def group_efeatures(
        cells,
        targets,
        use_global_rheobase=True,
        protocol_mode='mean'
):
    """
    Group the recordings and their efeatures and associate them to the
    EFeature Targets and Protocols they belong to.

    Args:
        cells (list): list of Cells containing for which the rheobase will be
            computed
        targets (dict): define the efeatures to extract as well as which
            protocols and current amplitude they should be extracted for. Of
            the form:
            [{
                "efeature": "AP_amplitude",
                "protocol": "IDRest",
                "amplitude": 150.,
                "tolerance": 10.,
                "efel_settings": {
                    'stim_start': 200.,
                    'stim_end': 500.,
                    'Threshold': -10.
                }
            }]
        use_global_rheobase (bool): As the final amplitude of a target is the
            mean of the amplitude of the cells, a global rheobase can be used
            to avoid issues when a cell matches a target but not another one.
            Which can result in situations where the second target is higher
            but it's amp in amperes is lower.
            e.g: target1 = 100%, target2 = 150% but target1_amp = 0.2 and
            target2_amp = 0.18
        protocol_mode (mean): if a protocol matches several recordings, the
            mode set the logic of how the output will be generating. Must be
            'mean', 'median' or 'lnmc'
    """

    global_rheobase = None
    if use_global_rheobase:
        global_rheobase = numpy.nanmean(
            [c.rheobase for c in cells if c.rheobase is not None]
        )

    protocols = _build_protocols(
        targets,
        global_rheobase=global_rheobase,
        protocol_mode=protocol_mode
    )

    for protocol in protocols:
        for cell in cells:

            if cell.rheobase is None:
                continue

            for recording in cell.get_recordings_by_protocol_name(
                    protocol.name
            ):

                if recording.in_target(
                        protocol.amplitude,
                        protocol.tolerance
                ):
                    protocol.append(recording)

    return protocols


def _build_current_dict(cells):
    """
    Compute the mean and standard deviation of the holding and threshold
    currents.
    """

    holding = {}
    threshold = {}

    for cell in cells:

        holding[cell.name] = numpy.nanmean(
            [t.hypamp for t in cell.recordings]
        )

        if cell.rheobase is not None:
            threshold[cell.name] = cell.rheobase

    currents = {
        "holding_current": [
            numpy.nanmean(list(holding.values())),
            numpy.nanstd(list(holding.values())),
        ],
        "threshold_current": [
            numpy.nanmean(list(threshold.values())),
            numpy.nanstd(list(threshold.values())),
        ],
        "all_holding_current": holding,
        "all_threshold_current": threshold,
    }

    return currents


def create_feature_protocol_files(
        cells,
        protocols,
        output_directory=None,
        threshold_nvalue_save=1,
        write_files=True,
        save_files_used=False,
):
    """
    Save the efeatures and protocols for each protocol/target combo
    in json file.

    Args:
        cells (list): list of Cells.
        protocols (list): list of Protocols.
        output_directory (str): path of the directory to which the features,
            protocols and currents will be saved.
        threshold_nvalue_save (int): minimum number of values needed for
            an efeatures to be averaged and returned in the output.
        write_files (bool): if True, the efeatures, protocols and currents
            will be saved in .json files in addition of being returned.
        save_files_used (bool): if True, the name of the recording files used
            in the computation of the features will be added to the
            efeatures.

    Returns:
        feat (dict)
        stim (dict)
        currents (dict)

    """

    out_features = {}
    out_stimuli = {}

    for protocol in protocols:

        stimname = protocol.stimulus_name

        tmp_feat = []

        for target in protocol.feature_targets:

            if target.sample_size < threshold_nvalue_save:
                logger.warning(
                    "Number of values < threshold_nvalue_save for efeature"
                    "{} stimulus {}. The efeature will be ignored"
                    "".format(target.efel_feature_name, stimname)
                )
                continue

            tmp_feat.append(target.as_legacy_dict(save_files_used))

        if not tmp_feat:
            logger.warning(
                "No efeatures for stimulus {}. The protocol will not "
                "be created.".format(stimname)
            )
            continue

        out_features[stimname] = {'soma': tmp_feat}
        out_stimuli[stimname] = protocol.as_dict()

    # Compute the mean and std of holding and threshold currents
    currents = _build_current_dict(cells)

    if write_files:

        if not output_directory:
            raise Exception(
                f"output_directory cannot be {output_directory}"
                f" if write_files is True."
            )

        _saving_data(
            output_directory,
            out_features,
            out_stimuli,
            currents
        )

    return out_features, out_stimuli, currents


def _read_extract(
        files_metadata,
        recording_reader,
        map_function,
        targets,
        efel_settings=None
):
    cells = read_recordings(
        files_metadata,
        recording_reader=recording_reader,
        map_function=map_function,
    )

    cells = extract_efeatures_at_targets(
        cells, targets, map_function=map_function, efel_settings=efel_settings
    )

    return cells


def _read_extract_low_memory(
        files_metadata, recording_reader, targets, efel_settings=None
):
    cells = []
    for cell_name in files_metadata:

        cell = read_recordings(
            {cell_name: files_metadata[cell_name]},
            recording_reader=recording_reader,
        )[0]

        cell.recordings = [rec for rec in cell.recordings if rec.amp > 0.]

        extract_efeatures_at_targets([cell], targets, efel_settings)

        # clean traces voltage and time
        for i in range(len(cell.recordings)):
            cell.recordings[i].t = None
            cell.recordings[i].voltage = None
            cell.recordings[i].current = None
            cell.recordings[i].reader_data = None

        cells.append(cell)
        gc.collect()

    return cells


def convert_legacy_targets(targets):
    """Convert targets of the form:
        protocol_name: {
                "amplitudes": [50, 100],
                "tolerances": [10, 10],
                "efeatures": {"Spikecount": {'Threshold': -10.}},
                "location": "soma"
            }
        }
    To ones of the form:
        [{
            "efeature": "AP_amplitude",
            "protocol": "IDRest",
            "amplitude": 150.,
            "tolerance": 10.,
            "efel_settings": {
                'stim_start': 200.,
                'stim_end': 500.,
                'Threshold': -10.
            }
        }]
    """

    formatted_targets = []

    for protocol_name, target in targets.items():
        for i, amplitude in enumerate(target["amplitudes"]):

            tolerances = target["tolerances"]
            if len(tolerances) == 1:
                tolerances = tolerances * len(target["amplitudes"])

            for efeature in target["efeatures"]:

                formatted_target = {
                    "efeature": efeature,
                    "protocol": protocol_name,
                    "amplitude": amplitude,
                    "tolerance": tolerances[i],
                    "efel_settings": {}
                }

                if isinstance(target["efeatures"], dict):
                    formatted_target["efel_settings"] = target[
                        "efeatures"][efeature]

                formatted_targets.append(formatted_target)

    return formatted_targets


def extract_efeatures(
        output_directory,
        files_metadata,
        targets,
        threshold_nvalue_save,
        protocols_rheobase=None,
        recording_reader=None,
        map_function=map,
        write_files=False,
        plot=False,
        low_memory_mode=False,
        spike_threshold_rheobase=1,
        protocol_mode="mean",
        efel_settings=None
):
    """
    Extract efeatures.

    Args:
        output_directory (str): path to the output directory
        files_metadata (dict): define for which cell and protocol each file
            has to be used. Of the form:
            {
                cell_id: {
                    protocol_name: [{file_metadata1}, {file_metadata1}]
                }
            }
            The entries required in the file_metadata are specific to each
            recording_reader (see bluepyemodel/reader.py to know which one are
            needed for your recording_reader).
            As the file_metadata contain file paths, a same file path might
            need to be present in the file metadata of different protocols if
            the path contains data coming from different stimuli (eg: for NWB).
        targets (list): define the efeatures to extract as well as which
            protocols and current amplitude they should be extracted for. Of
            the form:
            [{
                "efeature": "AP_amplitude",
                "protocol": "IDRest",
                "amplitude": 150.,
                "tolerance": 10.,
                "efel_settings": {
                    'stim_start': 200.,
                    'stim_end': 500.,
                    'Threshold': -10.
                }
            }]
        threshold_nvalue_save (int): minimum number of values needed for
            an efeatures to be averaged and returned in the output.
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        recording_reader (function): custom recording reader function. It's
            inner working has to match the metadata entered in files_metadata.
        map_function (function): Function used to map (parallelize) the
            recording reading and feature extraction operations.
        write_files (bool): if True, the efeatures, protocols and currents
            will be saved in .json files in addition of being returned.
        plot (bool): if True, the recordings and efeatures plots will be
            created.
        low_memory_mode (bool): if True, minimizes the amount of memory used
            during the data reading and feature extraction steps by performing
            additional clean up. Not compatible with map_function.
        spike_threshold_rheobase (int): number of spikes above which a
            recording is considered to compute the rheobase.
        protocol_mode (str): protocol_mode (mean): if a protocol matches
            several recordings, the mode set the logic of how the output
            will be generating. Must be 'mean', 'median' or 'lnmc'
        efel_settings (dict): efel settings in the form
            {setting_name: setting_value}. If settings are also informed
            in the targets per efeature, the latter will have priority.
    """

    if protocols_rheobase is None:
        protocols_rheobase = []

    if low_memory_mode and map_function != map:
        logger.warning(
            "low_memory_mode is not compatible with the use of map_function"
        )

    if low_memory_mode and plot:
        raise Exception('plot cannot be used in low_memory_mode mode.')

    if isinstance(targets, dict):
        logger.warning(
            "targets seems to be in a legacy format. A conversion will"
            " be performed."
        )
        targets = convert_legacy_targets(targets)

    if not low_memory_mode:
        cells = _read_extract(
            files_metadata,
            recording_reader,
            map_function,
            targets,
            efel_settings
        )
    else:
        cells = _read_extract_low_memory(
            files_metadata, recording_reader, targets, efel_settings
        )

    if protocols_rheobase:
        compute_rheobase(
            cells,
            protocols_rheobase=protocols_rheobase,
            spike_threshold=spike_threshold_rheobase
        )

    protocols = group_efeatures(
        cells,
        targets,
        use_global_rheobase=True,
        protocol_mode=protocol_mode
    )

    efeatures, protocol_definitions, current = create_feature_protocol_files(
        cells,
        protocols,
        output_directory=output_directory,
        threshold_nvalue_save=threshold_nvalue_save,
        write_files=write_files,
    )

    if plot:
        plot_all_recordings_efeatures(
            cells, protocols, output_dir=output_directory
        )

    return efeatures, protocol_definitions, current


def plot_recordings(
        files_metadata,
        output_directory="./figures/",
        recording_reader=None,
        map_function=map,
):
    """
    Plots recordings.

    Args:

        files_metadata (dict): define for which cell and protocol each file
            has to be used. Of the form:
            {
                cell_id: {
                    protocol_name: [{file_metadata1}, {file_metadata1}]
                }
            }
            A same file path might be present in the file metadata for
            different protocols.
            The entries required in the file_metadata are specific to each
            recording_reader (see bluepyemodel/reader.py to know which one
            are needed for your recording_reader).
        output_directory (str): path to the output directory where the plots
            will be saved.
        recording_reader (function): custom recording reader function. It's
            inner working has to match the metadata entered in files_metadata
        map_function (function): Function used to map (parallelize) the
            recording reading and feature extraction operations.
    """

    cells = read_recordings(
        files_metadata,
        recording_reader=recording_reader,
        map_function=map_function,
    )

    plot_all_recordings(cells, output_dir=output_directory)
