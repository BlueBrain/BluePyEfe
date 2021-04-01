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


def _extract_efeatures_cell(cell, targets, threshold):
    """
    Compute the efeatures on all the recordings of a Cell.

    The present function exists to be use by the map_function.
    """

    for prot_name, target in targets.items():
        cell.extract_efeatures(prot_name, target["efeatures"], threshold)

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
    cells, targets, threshold=-20.0, map_function=map
):
    """
    Extract efeatures from recordings following the protocols, amplitudes and
    efeature names specified in the targets.

    Args:
        cells (list): list of Cells containing the recordings from which the
            efeatures will be extracted.
        targets (dict): define the efeatures to extract for each protocols
            and the amplitude around which these features should be
            averaged. Of the form:
            {
                protocol_name: {
                    "amplitudes": [50, 100],
                    "tolerances": [10, 10],
                    "efeatures": ["Spikecount", "AP_amplitude"],
                    "location": "soma"
                }
            }
            If efeatures must be computed only for a given time interval,
            the beginning and end of this interval can be specified as
            follows (in ms):
            "efeatures": {
                "Spikecount": [500, 1100],
                "AP_amplitude": [100, 600],
            }
        threshold (float): voltage threshold (in mV) used for spike
        detection.
        map_function (function): Function used to map (parallelize) the
            feature extraction operations. Note: the parallelization is
            done across cells an not across efeatures.
    """

    for prot_name, target in targets.items():

        if len(target["tolerances"]) == 1:

            targets[prot_name]["tolerances"] = target["tolerances"] * len(
                target["amplitudes"]
            )

        if isinstance(target["efeatures"], list):
            targets[prot_name]["efeatures"] = {
                k: None for k in target["efeatures"]
            }

    cells = map_function(
        functools.partial(
            _extract_efeatures_cell,
            targets=targets,
            threshold=threshold,
        ),
        cells,
    )

    return list(cells)


def compute_rheobase(cells, protocols_rheobase):
    """
    For each cell, finds the smallest current inducing a spike (rheobase).
    This currents are then use it to compute the relative amplitude of
    the stimuli.

    Args:
        cells (list): list of Cells containing for which the rheobase will be
            computed
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
    """

    for cell in cells:
        cell.compute_rheobase(protocols_rheobase)
        cell.compute_relative_amp()


def mean_efeatures(cells, targets, use_global_rheobase=True):
    """
    Group the recordings and their efeatures and associate them to the
    Protocol (target) they belong with.

    Args:
        cells (list): list of Cells containing for which the rheobase will be
            computed
        targets (dict): see docstring of extract_efeatures.
        use_global_rheobase (bool): As the final amplitude of a target is the
            mean of the amplitude of the cells, a global rheobase can be used
            to avoid issues when a cell matches a target but not another one.
            Which can result in situations where the second target is higher
            but it's amp in amperes is lower.
            e.g: target1 = 100%, target2 = 150% but target1_amp = 0.2 and
            target2_amp = 0.18
    """

    global_rheobase = None
    if use_global_rheobase:
        global_rheobase = numpy.nanmean(
            [c.rheobase for c in cells if c.rheobase is not None]
        )

    protocols = []
    for protocol_name, target in targets.items():
        for i, amplitude in enumerate(target["amplitudes"]):

            if not type(amplitude) == int and not type(amplitude) == float:
                raise Exception(
                    "Target amplitudes have to be numbers, not {}".format(
                        type(amplitude)
                    )
                )

            protocol = Protocol(
                name=protocol_name,
                amplitude=amplitude,
                tolerance=target["tolerances"][i],
                efeatures=target["efeatures"],
                location=target["location"],
            )

            protocols.append(protocol)

    for protocol in protocols:
        for cell in cells:

            # Ignore the cells for which we couldn't compute
            # the rheobase
            if cell.rheobase is None:
                continue

            for recording in cell.get_recordings_by_protocol_name(
                protocol.name
            ):

                if recording.in_target(
                        protocol.amplitude,
                        protocol.tolerance
                ):
                    protocol.recordings.append(recording)

        # Update the ecode of protocol with the mean of the ecode of each
        # recording it contains.
        protocol.mean_ecode_params(global_rheobase)

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

    feat = {}
    stim = {}

    for prot in protocols:

        stimname = prot.name + "_" + str(prot.amplitude)

        for efeature in prot.efeatures:

            n = len(prot.get_efeature(efeature))
            m, s = prot.mean_std_efeature(efeature)

            if n < threshold_nvalue_save:
                logger.warning(
                    "Number of values < threshold_nvalue_save for efeature"
                    "{} stimulus {}. The efeature will be ignored"
                    "".format(efeature, stimname)
                )
                continue

            if numpy.isnan(m):
                logger.warning(
                    "Efeatures {} for stimulus {} is NaN and will be "
                    "ignored".format(efeature, stimname)
                )
                continue

            if s == 0.0:
                logger.warning(
                    "Standard deviation for efeatures {} for stimulus {} "
                    "is 0. and will be set to 1e-3".format(efeature, stimname)
                )
                s += 1e-3

            feature_definition = {"feature": efeature, "val": [m, s], "n": n}

            if stimname not in feat:
                feat[stimname] = {}
                feat[stimname][prot.location] = [feature_definition]
            else:
                feat[stimname][prot.location].append(feature_definition)

            if save_files_used:
                feat[stimname][prot.location][-1][
                    "files"
                ] = prot.get_files_used()

        if stimname in feat:

            if len(feat[stimname]):
                stim[stimname] = prot.ecode_params()
            else:
                feat.pop(stimname, None)
                logger.warning(
                    "No efeatures for stimulus {}. The protocol will not "
                    "be returned.".format(stimname)
                )

    # Compute the mean and std of holding and threshold currents
    currents = _build_current_dict(cells)

    if write_files:

        if not output_directory:
            raise Exception(
                f"output_directory cannot be {output_directory}"
                f" if write_files is True."
            )

        _saving_data(output_directory, feat, stim, currents)

    return feat, stim, currents


def _read_extract(
        files_metadata, recording_reader, map_function, targets, ap_threshold
):
    cells = read_recordings(
        files_metadata,
        recording_reader=recording_reader,
        map_function=map_function,
    )

    extract_efeatures_at_targets(
        cells, targets, threshold=ap_threshold, map_function=map_function
    )

    return cells


def _read_extract_low_memory(
        files_metadata, recording_reader, targets, ap_threshold
):

    cells = []
    for cell_name in files_metadata:

        cell = read_recordings(
            {cell_name: files_metadata[cell_name]},
            recording_reader=recording_reader,
        )[0]

        cell.recordings = [rec for rec in cell.recordings if rec.amp > 0.]

        extract_efeatures_at_targets(
            [cell], targets, threshold=ap_threshold
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


def extract_efeatures(
    output_directory,
    files_metadata,
    targets,
    threshold_nvalue_save,
    protocols_rheobase=[],
    ap_threshold=-20.0,
    recording_reader=None,
    map_function=map,
    write_files=False,
    plot=False,
    low_memory_mode=False,
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
        targets (dict): define the efeatures to extract for each protocols
            and the amplitude around which these features should be
            averaged. Of the form:
            {
                protocol_name: {
                    "amplitudes": [50, 100],
                    "tolerances": [10, 10],
                    "efeatures": ["Spikecount", "AP_amplitude"],
                    "location": "soma"
                }
            }
            If efeatures must be computed only for a given time interval,
            the beginning and end of this interval can be specified as
            follows (in ms):
            "efeatures": {
                "Spikecount": [500, 1100],
                "AP_amplitude": [100, 600],
            }
        threshold_nvalue_save (int): minimum number of values needed for
            an efeatures to be averaged and returned in the output.
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        ap_threshold (float): voltage threshold (in mV) used for spike
            detection.
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
    """

    if low_memory_mode and map_function != map:
        logger.warning(
            "low_memory_mode is not compatible with the use of map_function"
        )

    if low_memory_mode and plot:
        raise Exception('plot cannot be used in low_memory_mode mode.')

    if not(low_memory_mode):
        cells = _read_extract(
            files_metadata, recording_reader, map_function, targets,
            ap_threshold
        )
    else:
        cells = _read_extract_low_memory(
            files_metadata, recording_reader, targets, ap_threshold
        )

    if protocols_rheobase:
        compute_rheobase(cells, protocols_rheobase=protocols_rheobase)

    protocols = mean_efeatures(cells, targets, use_global_rheobase=True)

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
