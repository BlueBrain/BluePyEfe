"""Protocol class"""

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

import numpy
import logging

from bluepyefe.ecode import eCodes

logger = logging.getLogger(__name__)


class Protocol():

    """Protocol informs about the current stimulus that was used to obtain
     efeatures at a given amplitude for a given protocol name. This class
     is mainly used to produce a description of the experimental protocol
     that can be used in BluePyOpt"""

    def __init__(
        self,
        name,
        amplitude,
        tolerance,
        feature_targets=None,
        global_rheobase=None,
        mode="mean",
    ):
        """Constructor

        Args:
            name (str): name of the protocol (ex: 'APWaveform')
            amplitude (float): amplitude of the current stimuli for the
                present protocol (expressed as a percentage of the
                threshold amplitude or in absolute current depending on the
                setting absolute_amplitude)
            tolerance (float): tolerance around the target amplitude in which
                an experimental recording will be seen as a hit during
                efeatures extraction (expressed as a percentage of the
                threshold amplitude or in absolute current depending on the
                setting absolute_amplitude)
            feature_targets (list): list of EFeatureTarget associated to the
                protocol
            global_rheobase (float): average rheobase across all cells
            mode (str): if the protocol matches several recordings, the mode
                set the logic of how the output will be generating. Must be
                'mean', 'median' or 'lnmc'
        """

        self.name = name
        self.amplitude = amplitude
        self.tolerance = tolerance

        self.feature_targets = feature_targets
        if self.feature_targets is None:
            self.feature_targets = []

        self.global_rheobase = global_rheobase
        self.mode = mode

        self.recordings = []

    @property
    def stimulus_name(self):
        """Name of the stimulus associated to the protocol"""

        return f"{self.name}_{self.amplitude}"

    @property
    def n_match(self):
        """Number of recordings whose amplitude matched the present protocol"""

        return sum([f.sample_size for f in self.feature_targets])

    @property
    def ecode(self):
        """Create a temporary eCode that matches all the recordings for the
        present protocol. The eCode's parameters are computed differently
        depending on the mode of the protocol"""

        if not self.recordings:
            return None

        for ecode in eCodes.keys():
            if ecode.lower() in self.name.lower():
                ecode = eCodes[ecode]({}, {}, self.name)
                break
        else:
            raise KeyError(
                "There is no eCode linked to the stimulus name {}. See "
                "ecode/__init__.py for the available stimuli names"
                "".format(self.name.lower())
            )

        if self.mode == "mean":
            self.reduce_ecode(ecode, operator=numpy.nanmean)
        elif self.mode == "median":
            self.reduce_ecode(ecode, operator=numpy.nanmedian)
        elif self.mode == "min":
            self.reduce_ecode(ecode, operator=numpy.nanmin)
        elif self.mode == "max":
            self.reduce_ecode(ecode, operator=numpy.nanmax)
        else:
            raise ValueError("'mode' should be mean or median")

        return ecode

    def append(self, recording):
        """Append a Recording to the present protocol"""

        for i, target in enumerate(self.feature_targets):
            if target.efeature_name in recording.efeatures:
                self.feature_targets[i].append(
                    recording.efeatures[target.efeature_name],
                    recording.files
                )

            if (
                recording.auto_threshold is not None and
                "Threshold" not in self.feature_targets[i].efel_settings
            ):
                self.feature_targets[i]._auto_thresholds.append(
                    recording.auto_threshold)

        self.recordings.append(recording)

    def as_dict(self):
        """Returns a dictionary that defines the present protocol. This
        definition is computed differently depending on the mode of the
        protocol
        """

        return {
            "holding": {
                "delay": 0.0,
                "amp": self.ecode.hypamp,
                "duration": self.ecode.tend,
                "totduration": self.ecode.tend,
            },
            "step": self.ecode.get_stimulus_parameters(),
        }

    def reduce_ecode(self, ecode, operator):
        """Creates an eCode defined from the parameters of all the recordings
        matching the present protocol"""

        if not self.recordings:
            logger.warning(
                "Could not compute average ecode for protocol {} target {} "
                "because it didn't match any recordings".format(
                    self.name, self.amplitude
                )
            )
            return None

        params = [r.get_params() for r in self.recordings]

        if self.global_rheobase is None and "amp" in params[0]:
            logger.warning(
                "No global threshold amplitude passed. This can result"
                " in inconsistencies in-between protocols if some cells "
                "only matched a subset of the targets."
            )

        for key in params[0]:

            if isinstance(params[0][key], (list, numpy.ndarray)):
                logger.warning(
                    "Parameter {} for protocol {} is a list and cannot be "
                    "averaged across recordings".format(key, self.name)
                )
                setattr(ecode, key, params[0][key])
                continue

            if key == "amp" and self.global_rheobase:
                amp_rel = operator([c["amp_rel"] for c in params])
                mean_param = float(amp_rel) * self.global_rheobase / 100.
            elif key == "amp2" and self.global_rheobase:
                amp_rel2 = operator([c["amp2_rel"] for c in params])
                mean_param = float(amp_rel2) * self.global_rheobase / 100.
            else:
                mean_param = operator([numpy.nan if c[key] is None else c[key] for c in params])

            if numpy.isnan(mean_param):
                mean_param = None

            setattr(ecode, key, mean_param)

        return ecode

    def __str__(self):
        """String representation"""

        str_form = "Protocol {} {:.1f}%:\n".format(
            self.name, self.amplitude
        )

        str_form += "Number of matching recordings: {}".format(self.n_match)

        if self.n_match:
            str_form += "\neCode: {}\n".format(self.as_dict)

        return str_form
