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
import logging
import numpy
import matplotlib.pyplot as plt
import pathlib

from bluepyefe.ecode import eCodes
from bluepyefe.reader import *
from bluepyefe.plotting import _save_fig

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
            return nwb_reader(config_data)

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
        self,
        protocol_data,
        protocol_name,
        recording_reader=None,
        efel_settings=None
    ):
        """
        For each recording's metadata, instance a recording object and
        populate it by reading the matching data file.
        """

        for config_data in protocol_data:

            if "protocol_name" not in config_data:
                config_data["protocol_name"] = protocol_name

            for reader_data in self.reader(config_data, recording_reader):

                for ecode in eCodes.keys():
                    if ecode.lower() in protocol_name.lower():
                        rec = eCodes[ecode](
                            config_data,
                            reader_data,
                            protocol_name,
                            efel_settings
                        )
                        self.recordings.append(rec)
                        break
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
            efeature_names=None,
            efel_settings=None
    ):
        """
        Extract the efeatures for the recordings matching the protocol name.
        """

        for i in self.get_recordings_id_by_protocol_name(protocol_name):
            self.recordings[i].compute_efeatures(
                efeatures, efeature_names, efel_settings)

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

    def plot_recordings(self, protocol_name, output_dir=None, show=False):
        """Plot all the recordings matching a protocol name"""

        recordings = self.get_recordings_by_protocol_name(protocol_name)

        if not len(recordings):
            return None, None

        recordings_amp = [rec.amp for rec in recordings]
        recordings = [recordings[k] for k in numpy.argsort(recordings_amp)]

        n_cols = 10
        n_rows = int(2 * numpy.ceil(len(recordings) / n_cols))

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=[3.0 + 1.9 * int(n_cols), 2.4 * n_rows],
            squeeze=False
        )

        for i, rec in enumerate(recordings):

            col = i % int(n_cols)
            row = 2 * int(i / n_cols)

            display_ylabel = col == 0
            display_xlabel = row + 1 == axs.shape[0]

            _, _ = rec.plot(
                axis_current=axs[row][col],
                axis_voltage=axs[row + 1][col],
                display_xlabel=display_xlabel,
                display_ylabel=display_ylabel
            )

        fig.suptitle("Cell: {}, Experiment: {}".format(self.name, protocol_name))

        plt.subplots_adjust(wspace=0.53, hspace=0.7)

        for ax in axs.flatten():
            if not ax.lines:
                ax.set_visible(False)

        # Do not use tight-layout, it significantly increases the runtime
        plt.margins(0, 0)

        if show:
            fig.show()

        if output_dir is not None:
            filename = "{}_{}_recordings.pdf".format(self.name, protocol_name)
            dirname = pathlib.Path(output_dir) / self.name
            _save_fig(dirname, filename)

        return fig, axs

    def plot_all_recordings(self, output_dir=None, show=False):
        """Plot all the recordings of the cell"""

        for protocol_name in self.get_protocol_names():
            self.plot_recordings(protocol_name, output_dir, show=show)
