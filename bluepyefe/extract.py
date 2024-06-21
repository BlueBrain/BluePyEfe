"""Efeature extraction functions"""

"""
Copyright (c) 2022, EPFL/Blue Brain Project

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
import os
import pickle
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
from bluepyefe.rheobase import compute_rheobase_absolute
from bluepyefe.rheobase import compute_rheobase_flush
from bluepyefe.rheobase import compute_rheobase_majority_bin
from bluepyefe.rheobase import compute_rheobase_interpolation
from bluepyefe.tools import DEFAULT_EFEL_SETTINGS, PRESET_PROTOCOLS_RHEOBASE
from bluepyefe.auto_targets import default_auto_targets

logger = logging.getLogger(__name__)


def _create_cell(cell_definition, recording_reader, efel_settings=None):
    """
    Initialize a Cell object and populate it with the content of the associated
    recording files.

    The present function exist to be use by the map_function.

    Args:
        cell_definition (dict): subpart of files_metadata containing the
            information for a single cell.
        recording_reader (callable or None): method that will be used to
            read the files containing the recordings. If None, the function
            used will be chosen automatically based on the extension
            of the file.
        efel_settings (dict): eFEL settings in the form
            {setting_name: setting_value}.
    """

    cell_name = cell_definition[0]
    cell = cell_definition[1]

    out_cell = Cell(name=cell_name)

    for prot_name in cell.keys():
        out_cell.read_recordings(
            protocol_data=cell[prot_name],
            protocol_name=prot_name,
            recording_reader=recording_reader,
            efel_settings=efel_settings
        )

    return out_cell


def _extract_efeatures_cell(cell, targets, efel_settings=None):
    """
    Compute the efeatures on all the recordings of a Cell.

    The present function exists to be use by the map_function.

    Args:
        cell (Cell): cell for which to extract the efeatures.
        targets (list of Target): targets to extract from the recordings of
            the present cell.
        efel_settings (dict): eFEL settings in the form
            {setting_name: setting_value}.
    """

    if efel_settings is None:
        efel_settings = {}

    # Group targets per same protocol and same eFEL settings for efficiency
    setting_groups = []
    for target in targets:

        efeature_name = target.get("efeature_name", target["efeature"])

        for i, group in enumerate(setting_groups):

            if target["efel_settings"] == group['efel_settings'] and \
                    target["protocol"] == group['protocol']:
                setting_groups[i]["efeatures"].append(target["efeature"])
                setting_groups[i]['efeature_names'].append(efeature_name)
                break

        else:

            setting_group = {
                'efel_settings': target["efel_settings"],
                'protocol': target["protocol"],
                'efeatures': [target["efeature"]],
                'efeature_names': [efeature_name]
            }
            setting_groups.append(setting_group)

    for group in setting_groups:
        cell.extract_efeatures(
            group['protocol'],
            group["efeatures"],
            group["efeature_names"],
            efel_settings={**efel_settings, **group["efel_settings"]}
        )

    return cell


def _saving_data(output_directory, feat, stim, currents):
    """
    Save the features, protocols and current to json files.

    Args:
        output_directory (str): path of the directory to which the efeatures
            and protocols data will be saved.
        feat (dict): contains the information related to the efeatures.
        stim (dict): contains the information related to the protocols.
        currents (dict): contains the information related to the holding
            and threshold currents.
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


def read_recordings(
    files_metadata,
    recording_reader=None,
    map_function=map,
    efel_settings=None
):
    """
    Read recordings from a group of files. The files are expected to be
    identified by both a cell id and a protocol name (see files_metadata).

    Args:
        files_metadata (dict): define for which cell and protocol each file
            has to be used. Of the form:

                .. code-block:: python

                    {
                        cell_id: {
                            protocol_name: [
                                {file_metadata1},
                                {file_metadata1}
                            ]
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
        efel_settings (dict): eFEL settings in the form
            {setting_name: setting_value}.

    Return:
         cells (list): list of Cell objects containing the data of the
         recordings
    """

    cells = map_function(
        functools.partial(
            _create_cell,
            recording_reader=recording_reader,
            efel_settings=efel_settings
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

            .. code-block:: python

                [{
                    "efeature": "AP_amplitude",
                    "protocol": "IDRest",
                    "amplitude": 150.,
                    "tolerance": 10.,
                    "efel_settings": {
                        "stim_start": 200.,
                        "stim_end": 500.,
                        "Threshold": -10.
                    }
                }]

        map_function (function): Function used to map (parallelize) the
            feature extraction operations. Note: the parallelization is
            done across cells an not across efeatures.
        efel_settings (dict): eFEL settings in the form
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


def compute_rheobase(
    cells,
    protocols_rheobase,
    rheobase_strategy="absolute",
    rheobase_settings=None
):
    """
    For each cell, finds the smallest current inducing a spike (rheobase).
    This currents are then use it to compute the relative amplitude of
    the stimuli.

    Args:
        cells (list): list of Cells containing for which the rheobase will be
            computed
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        rheobase_strategy (str): function used to compute the rheobase. Can be
            'absolute' (amplitude of the lowest amplitude inducing at least a
            spike) or 'majority' (amplitude of the bin in which a majority of
            sweeps induced at least one spike).
        rheobase_settings (dict): settings related to the rheobase computation.
            Keys have to match the arguments expected by the rheobase
            computation function.
    """

    if rheobase_settings is None:
        rheobase_settings = {}

    if rheobase_strategy == "absolute":
        rheobase_function = compute_rheobase_absolute
    elif rheobase_strategy == "flush":
        rheobase_function = compute_rheobase_flush
    elif rheobase_strategy == "majority":
        rheobase_function = compute_rheobase_majority_bin
    elif rheobase_strategy == "interpolation":
        rheobase_function = compute_rheobase_interpolation
    else:
        raise Exception(f"Rheobase strategy {rheobase_strategy} unknown.")

    for cell in cells:
        rheobase_function(cell, protocols_rheobase, **rheobase_settings)
        cell.compute_relative_amp()


def _build_protocols(
    targets,
    global_rheobase,
    protocol_mode,
    efel_settings=None
):
    """Build a list of Protocols that matches the expected targets"""

    if efel_settings is None:
        efel_settings = {}

    protocols = []

    for target in targets:

        settings = {**efel_settings, **target.get('efel_settings', {})}
        efeature_name = target.get("efeature_name", target["efeature"])

        efeature_target = EFeatureTarget(
            efeature_name=efeature_name,
            efel_feature_name=target['efeature'],
            protocol_name=target['protocol'],
            amplitude=target['amplitude'],
            tolerance=target['tolerance'],
            efel_settings=settings,
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
    absolute_amplitude=False,
    use_global_rheobase=True,
    protocol_mode='mean',
    efel_settings=None
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

            .. code-block:: python

                [{
                    "efeature": "AP_amplitude",
                    "protocol": "IDRest",
                    "amplitude": 150.,
                    "tolerance": 10.,
                    "efel_settings": {
                        "stim_start": 200.,
                        "stim_end": 500.,
                        "Threshold": -10.
                    }
                }]

        absolute_amplitude (bool): if True, will use the absolute amplitude
            instead of the relative amplitudes of the recordings when checking
            if a recording has to be used for a given target.
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
        efel_settings (dict): eFEL settings in the form
            {setting_name: setting_value}. If settings are also informed
            in the targets per efeature, the latter will have priority.
    """

    if efel_settings is None:
        efel_settings = {}

    global_rheobase = None
    if use_global_rheobase and not absolute_amplitude:
        global_rheobase = numpy.nanmean(
            [c.rheobase for c in cells if c.rheobase is not None]
        )

    protocols = _build_protocols(
        targets,
        global_rheobase=global_rheobase,
        protocol_mode=protocol_mode,
        efel_settings=efel_settings,
    )

    for protocol in protocols:
        for cell in cells:

            if cell.rheobase is None and not absolute_amplitude:
                continue

            for recording in cell.get_recordings_by_protocol_name(
                    protocol.name
            ):

                if recording.in_target(
                    protocol.amplitude,
                    protocol.tolerance,
                    absolute_amplitude
                ):
                    protocol.append(recording)

    return protocols


def _build_current_dict(cells, default_std_value):
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

    std_holding = numpy.nanstd(list(holding.values()))
    if std_holding == 0:
        std_holding = default_std_value

    std_threshold = numpy.nanstd(list(threshold.values()))
    if std_threshold == 0:
        std_threshold = default_std_value

    currents = {
        "holding_current": [
            numpy.nanmean(list(holding.values())),
            std_holding,
        ],
        "threshold_current": [
            numpy.nanmean(list(threshold.values())),
            std_threshold,
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
    default_std_value=1e-3
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
        default_std_value (float): default value used to replace the standard
            deviation if the standard deviation is 0.

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
                    "Number of values < threshold_nvalue_save for efeature "
                    "{} stimulus {}. The efeature will be ignored"
                    "".format(target.efel_feature_name, stimname)
                )
                continue

            tmp_feat.append(target.as_dict(save_files_used, default_std_value))

        if not tmp_feat:
            logger.warning(
                "No efeatures for stimulus {}. The protocol will not "
                "be created.".format(stimname)
            )
            continue

        out_features[stimname] = {'soma': tmp_feat}
        out_stimuli[stimname] = protocol.as_dict()

    # Compute the mean and std of holding and threshold currents
    currents = _build_current_dict(cells, default_std_value)

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
    """Read recordings and create the matching Cell objects based on a files_metadata."""

    cells = read_recordings(
        files_metadata,
        recording_reader=recording_reader,
        map_function=map_function,
        efel_settings=efel_settings
    )

    cells = extract_efeatures_at_targets(
        cells, targets, map_function=map_function, efel_settings=efel_settings
    )

    return cells


def _read_extract_low_memory(
    files_metadata, recording_reader, targets, efel_settings=None
):
    """Read recordings and create the matching Cell objects based on a
    files_metadata. Does not us a map function and delete the recording's
    data on the go to avoid using too much memory."""

    cells = []
    for cell_name in files_metadata:

        cell = read_recordings(
            {cell_name: files_metadata[cell_name]},
            recording_reader=recording_reader,
            efel_settings=efel_settings
        )[0]

        extract_efeatures_at_targets(
            [cell], targets, map, efel_settings
        )

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
    """Convert targets of the form::

        .. code-block:: python

            protocol_name: {
                "amplitudes": [50, 100],
                "tolerances": [10, 10],
                "efeatures": {"Spikecount": {'Threshold': -10.}},
                "location": "soma"
            }

    To ones of the form::

        .. code-block:: python

            [{
                "efeature": "AP_amplitude",
                "protocol": "IDRest",
                "amplitude": 150.,
                "tolerance": 10.,
                "efel_settings": {
                    "stim_start": 200.,
                    "stim_end": 500.,
                    "Threshold": -10.
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


def _extract_with_targets(
    files_metadata,
    targets=None,
    protocols_rheobase=None,
    absolute_amplitude=False,
    recording_reader=None,
    map_function=map,
    low_memory_mode=False,
    protocol_mode="mean",
    efel_settings=None,
    rheobase_strategy="absolute",
    rheobase_settings=None
):
    """Read the recordings and extract the efeatures at the requested
    targets"""

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

    if not absolute_amplitude:
        compute_rheobase(
            cells,
            protocols_rheobase=protocols_rheobase,
            rheobase_strategy=rheobase_strategy,
            rheobase_settings=rheobase_settings
        )

    protocols = group_efeatures(
        cells,
        targets,
        absolute_amplitude=absolute_amplitude,
        use_global_rheobase=True,
        protocol_mode=protocol_mode,
        efel_settings=efel_settings
    )

    return cells, protocols


def _extract_auto_targets(
    files_metadata,
    protocols_rheobase,
    recording_reader,
    map_function,
    protocol_mode,
    efel_settings,
    auto_targets=None,
    rheobase_strategy="flush",
    rheobase_settings=None
):
    """Read the recordings and extract the efeatures using AutoTargets"""

    if rheobase_settings is None and rheobase_strategy == "flush":
        rheobase_settings = {"upper_bound_spikecount": 4}

    if auto_targets is None:
        auto_targets = default_auto_targets()

    cells = read_recordings(
        files_metadata,
        recording_reader=recording_reader,
        map_function=map_function,
        efel_settings=efel_settings
    )

    compute_rheobase(
        cells,
        protocols_rheobase=protocols_rheobase,
        rheobase_strategy=rheobase_strategy,
        rheobase_settings=rheobase_settings
    )

    if not sum(bool(c.rheobase) for c in cells):
        logger.warning("No cells with valid rheobase")
        return cells, [], []

    recordings = []
    for c in cells:
        recordings += c.recordings

    for i in range(len(auto_targets)):
        auto_targets[i].select_ecode_and_amplitude(recordings)

    # Extract the efeatures and group them around the preset of targets.
    targets = []
    for at in auto_targets:
        targets += at.generate_targets()

    cells = extract_efeatures_at_targets(
        cells, targets, map_function=map_function, efel_settings=efel_settings
    )

    protocols = group_efeatures(
        cells,
        targets,
        use_global_rheobase=True,
        protocol_mode=protocol_mode,
        efel_settings=efel_settings
    )

    return cells, protocols, targets


def extract_efeatures_per_cell(
    files_metadata,
    cells,
    output_directory,
    targets,
    protocol_mode,
    threshold_nvalue_save,
    write_files,
    default_std_value=1e-3
):

    for cell_name in files_metadata:

        cell_directory = str(pathlib.Path(output_directory) / cell_name)

        for cell in cells:

            if cell.name != cell_name:
                continue

            cell_protocols = group_efeatures(
                [cell],
                targets,
                use_global_rheobase=True,
                protocol_mode=protocol_mode
            )

            _ = create_feature_protocol_files(
                [cell],
                cell_protocols,
                output_directory=cell_directory,
                threshold_nvalue_save=threshold_nvalue_save,
                write_files=write_files,
                default_std_value=default_std_value
            )


def extract_efeatures(
    output_directory,
    files_metadata,
    targets=None,
    threshold_nvalue_save=1,
    protocols_rheobase=None,
    absolute_amplitude=False,
    recording_reader=None,
    map_function=map,
    write_files=False,
    plot=False,
    low_memory_mode=False,
    protocol_mode="mean",
    efel_settings=None,
    extract_per_cell=False,
    rheobase_strategy="absolute",
    rheobase_settings=None,
    auto_targets=None,
    pickle_cells=False,
    default_std_value=1e-3
):
    """
    Extract efeatures.

    Args:
        output_directory (str): path to the output directory
        files_metadata (dict): define from files to read the data as well as
            the name of the cells and protocols to which these data are
            related. Of the form::

                .. code-block:: python

                    {
                        cell_id: {
                            protocol_name: [
                                {file_metadata1},
                                {file_metadata1}
                            ]
                        }
                    }

            The entries required in the file_metadata are specific to each
            recording_reader (see bluepyemodel/reader.py to know which one are
            needed for your recording_reader).
            As the file_metadata contain file paths, a same file path might
            need to be present in the file metadata of different protocols if
            the path contains data coming from different stimuli (eg: for NWB).
        targets (list): define the efeatures to extract as well as which
            protocols and current amplitude (expressed either in % of the
            rheobase if absolute_amplitude if False or in nA if
            absolute_amplitude is True) they should be extracted.
            If targets are not provided, automatic targets will be used.
            Of the form::

                .. code-block:: python

                    [{
                        "efeature": "AP_amplitude",
                        "protocol": "IDRest",
                        "amplitude": 150.,
                        "tolerance": 10.,
                        "efel_settings": {
                            "stim_start": 200.,
                            "stim_end": 500.,
                            "Threshold": -10.
                        }
                    }]

        threshold_nvalue_save (int): minimum number of values needed for
            an efeatures to be averaged and returned in the output.
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        absolute_amplitude (bool): if True, will use the absolute amplitude
            instead of the relative amplitudes of the recordings when checking
            if a recording has to be used for a given target.
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
        protocol_mode (str): protocol_mode (mean): if a protocol matches
            several recordings, the mode set the logic of how the output
            will be generating. Must be 'mean', 'median' or 'lnmc'
        efel_settings (dict): eFEL settings in the form
            {setting_name: setting_value}. If settings are also informed
            in the targets per efeature, the latter will have priority. If
            None, will be set to::

                .. code-block:: python

                    {
                        "strict_stiminterval": True,
                        "Threshold": -20.,
                        "interp_step": 0.025
                    }

        extract_per_cell (bool): if True, also generates the features.json and
            protocol.json for each individual cells.
        rheobase_strategy (str): function used to compute the rheobase. Can be
            'absolute' (amplitude of the lowest amplitude inducing at least a
            spike) or 'majority' (amplitude of the bin in which a majority of
            sweeps induced at least one spike).
        rheobase_settings (dict): settings related to the rheobase computation.
            Keys have to match the arguments expected by the rheobase
            computation function.
        auto_targets (list of AutoTarget): targets with more flexible goals.
        pickle_cells (bool): if True, the cells object will be saved as a pickle file.
        default_std_value (float): default value used to replace the standard
            deviation if the standard deviation is 0.
    """

    if not files_metadata:
        raise ValueError("Argument 'files_metadata' is empty")

    if efel_settings is None:
        logger.warning(
            "efel_settings is None. Default settings will be used: " +
            str(DEFAULT_EFEL_SETTINGS)
        )
        efel_settings = DEFAULT_EFEL_SETTINGS.copy()

    if protocols_rheobase is None and not absolute_amplitude:
        logger.warning(
            "protocols_rheobase is None. Default protocol names will be used"
        )
        protocols_rheobase = PRESET_PROTOCOLS_RHEOBASE.copy()

    if low_memory_mode and map_function not in [map, None]:
        logger.warning(
            "low_memory_mode is not compatible with the use of map_function"
        )
    if low_memory_mode and plot:
        raise Exception('plot cannot be used in low_memory_mode mode.')

    if targets is not None and isinstance(targets, dict):
        logger.warning(
            "targets seems to be in a legacy format. A conversion will"
            " be performed."
        )
        targets = convert_legacy_targets(targets)

    if targets is not None and auto_targets is not None:
        raise Exception("Cannot specify both targets and auto_targets.")

    if (
        not absolute_amplitude and
        (targets is None or auto_targets is not None)
    ):
        cells, protocols, targets = _extract_auto_targets(
            files_metadata,
            protocols_rheobase,
            recording_reader,
            map_function,
            protocol_mode,
            efel_settings,
            auto_targets,
            rheobase_strategy,
            rheobase_settings
        )
    else:
        cells, protocols = _extract_with_targets(
            files_metadata,
            targets,
            protocols_rheobase,
            absolute_amplitude,
            recording_reader,
            map_function,
            low_memory_mode,
            protocol_mode,
            efel_settings,
            rheobase_strategy,
            rheobase_settings
        )

    efeatures, protocol_definitions, current = create_feature_protocol_files(
        cells,
        protocols,
        output_directory=output_directory,
        threshold_nvalue_save=threshold_nvalue_save,
        write_files=write_files,
        default_std_value=default_std_value
    )

    if pickle_cells:
        path_cells = pathlib.Path(output_directory)
        path_cells.mkdir(parents=True, exist_ok=True)
        pickle.dump(cells, open(path_cells / "cells.pkl", 'wb'))
        pickle.dump(protocols, open(path_cells / "protocols.pkl", 'wb'))

    if plot:
        plot_all_recordings_efeatures(
            cells, protocols, output_dir=output_directory, mapper=map_function
        )

    if extract_per_cell and write_files:
        extract_efeatures_per_cell(
            files_metadata,
            cells,
            output_directory,
            targets,
            protocol_mode,
            threshold_nvalue_save,
            write_files,
        )

    if not efeatures or not protocol_definitions:
        logger.warning("The output of the extraction is empty. Something went "
                       "wrong. Please check that your targets, files_metadata "
                       "and protocols_rheobase match the data you have "
                       "available.")

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

                .. code-block:: python

                    {
                        cell_id: {
                            protocol_name: [
                                {file_metadata1},
                                {file_metadata1}
                            ]
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
