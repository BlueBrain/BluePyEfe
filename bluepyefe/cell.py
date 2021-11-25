"""Cell class"""

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
from bluepyefe.ecode import eCodes
from bluepyefe.reader import *

import numpy as np

logger = logging.getLogger(__name__)


class Cell(object):

    """Contains the metadata related to the cell as well as the recordings data
    once these are read"""

    def __init__(self, name):
        """
        Constructor

        Args:
            name (str): name of the cell.
            recording_reader (function): custom recording reader matching the
                files metadata.
        """

        self.name = name

        self.recordings = []
        self.rheobase = None

    def reader(self, config_data, recording_reader=None):

        if "v_file" in config_data:
            filename = config_data["v_file"]
        elif "filepath" in config_data:
            filename = config_data["filepath"]

        if recording_reader:
            return recording_reader(config_data)
        if ".abf" in filename:
            return axon_reader(config_data)
        if ".ibw" in filename or ".bwav" in filename:
            return igor_reader(config_data)
        if ".nwb" in filename:
            return nwb_reader_BBP(config_data)

        raise Exception(
            "The format of the files is unknown and no custom reader were"
            " provided."
        )

    def get_protocol_names(self):
        return list(set([rec.protocol_name for rec in self.recordings]))

    def get_recordings_by_protocol_name(self, protocol_name):
        return [
            rec
            for rec in self.recordings
            if rec.protocol_name == protocol_name
        ]

    def get_recordings_id_by_protocol_name(self, protocol_name):
        return [
            i
            for i, trace in enumerate(self.recordings)
            if trace.protocol_name == protocol_name
        ]

    def read_recordings(
        self, protocol_data, protocol_name, recording_reader=None
    ):
        """
        For each recording's metadata, instance a recording object and
        populate it by reading the matching data file.
        """

        for config_data in protocol_data:
            for reader_data in self.reader(config_data, recording_reader):

                if protocol_name.lower() in eCodes.keys():
                    rec = eCodes[protocol_name.lower()](
                        config_data, reader_data, protocol_name
                    )

                    self.recordings.append(rec)
                else:
                    raise KeyError(
                        "There is no eCode linked to the stimulus name {}. "
                        "See ecode/__init__.py for the available stimuli "
                        "names".format(protocol_name.lower())
                    )

    def extract_efeatures(
            self,
            protocol_name,
            efeatures,
            efel_settings=None
    ):
        """
        Extract the efeatures for the recordings matching the protocol name.
        """

        for i in self.get_recordings_id_by_protocol_name(protocol_name):
            self.recordings[i].compute_efeatures(efeatures, efel_settings)

    def compute_rheobase(self, protocols_rheobase, spike_threshold=1, min_step=0.001,
                         majority=0.5):
        """
        Compute the rheobase by finding the smallest current amplitude
        triggering at least 'spike_threshold' spikes (default one) in the
        majority (default 50%) of the sweeps with a certain amplitude.

        Args:
            protocols_rheobase (list): names of the protocols that will be
                used to compute the rheobase of the cells. E.g: ['IDthresh'].
            spike_threshold (int): number of spikes above which a recording
                is considered to compute the rheobase.
            min_step (float): minimum step above which amplitudes can be
                considered as separate steps
            majority (float): the proportion of sweeps with spike_threshold
                spikes to consider the target amplitude as rheobase
        """
        amps = []
        spike_counts = []

        # collect all amps and spikecounts for the protocol_rheobase
        for i, rec in enumerate(self.recordings):
            if rec.protocol_name in protocols_rheobase:
                amps.append(rec.amp)
                spike_counts.append(rec.spikecount)

                if rec.amp < 0.01 and rec.spikecount >= spike_threshold:
                    logger.warning(
                        f"A recording of cell {self.name} protocol "
                        f"{rec.protocol_name} shows spikes at a "
                        "suspiciously low current in a trace from file"
                        f" {rec.files}. Check that the ton and toff are"
                        "correct or for the presence of unwanted spikes."
                    )

        # discretize values based on the min_step 
        amps_discrete = []
        spike_counts_discrete = []
        sort_idxs = np.argsort(amps)
        spike_counts_sorted = np.array(spike_counts)[sort_idxs]
        amps_sorted = np.array(amps)[sort_idxs]
        step_idxs = np.where(np.diff(amps_sorted) > min_step)[0] + 1

        for i, idx in enumerate(step_idxs):
            if i == 0:
                amp_median = np.median(amps_sorted[:idx])
                amps_discrete.append(amp_median)
                spike_counts_discrete.append(spike_counts_sorted[:idx])
            elif i < len(step_idxs):
                amp_median = np.median(amps_sorted[step_idxs[i - 1]:idx])
                amps_discrete.append(amp_median)
                spike_counts_discrete.append(spike_counts_sorted[step_idxs[i - 1]:idx])
        # append last
        amp_median = np.median(amps_sorted[step_idxs[-1]:])
        amps_discrete.append(amp_median)
        spike_counts_discrete.append(spike_counts_sorted[step_idxs[-1]:])

        # now for each value, count the number of spike_threshold
        for (amp, counts) in zip(amps_discrete, spike_counts_discrete):
            if len(counts) > 1:
                num_target = len(np.where(counts == spike_threshold)[0])
                if num_target / len(counts) > majority:
                    self.rheobase = amp
                    break
            else:
                if counts[0] == spike_threshold:
                    self.rheobase = amp
                    break

    def compute_relative_amp(self):
        """
        Compute the relative current amplitude for all the recordings as a
        percentage of the rheobase.
        """

        if self.rheobase not in (0.0, None, False, numpy.nan):

            for i in range(len(self.recordings)):
                self.recordings[i].compute_relative_amp(self.rheobase)

        else:

            logger.warning(
                "Cannot compute the relative current amplitude for the "
                "recordings of cell {} because its rheobase is {}."
                "".format(
                    self.name, self.rheobase
                )
            )
            self.rheobase = None
