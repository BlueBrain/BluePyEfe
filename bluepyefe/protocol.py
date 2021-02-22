"""Protocol class"""

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

from bluepyefe.ecode import eCodes

logger = logging.getLogger(__name__)


class Protocol(object):

    """ Protocol class """

    def __init__(
        self, name, amplitude, tolerance=20.0, efeatures=[], location="soma"
    ):

        self.name = name
        self.amplitude = amplitude
        self.efeatures = efeatures
        self.tolerance = tolerance
        self.location = location

        self.recordings = []

        if self.name.lower() in eCodes:
            self.ecode = eCodes[self.name.lower()]({}, {}, name)
        else:
            raise KeyError(
                "There is no eCode linked to the stimulus name {}. See "
                "ecode/__init__.py for the available stimuli names"
                "".format(self.name.lower())
            )

    def get_efeature(self, efeature):
        return [t.efeatures[efeature] for t in self.recordings]

    def mean_std_efeature(self, efeature):
        list_feature = self.get_efeature(efeature)
        return numpy.nanmean(list_feature), numpy.nanstd(list_feature)

    def mean_ecode_params(self, global_amp_threshold=None):

        if len(self.recordings):

            ecodes_params = [t.get_params() for t in self.recordings]

            if global_amp_threshold is None and "amp" in ecodes_params[0]:
                logger.warning(
                    "No global threshold amplitude passed. This can result"
                    " in inconsistencies in-between protocols if some cells "
                    "only matched a subset of the targets."
                )

            for key in ecodes_params[0]:

                if key == "amp" and global_amp_threshold is not None:
                    amp_rel = numpy.nanmean(
                        [c["amp_rel"] for c in ecodes_params]
                    )
                    param_mean = float(amp_rel) * global_amp_threshold / 100.0
                    setattr(self.ecode, key, param_mean)

                elif (
                    isinstance(ecodes_params[0][key], list)
                    or type(ecodes_params[0][key]) is numpy.ndarray
                ):
                    param_mean = numpy.asarray([c[key] for c in ecodes_params])
                    param_mean = numpy.nanmean(param_mean, axis=0)
                    setattr(self.ecode, key, param_mean)

                else:
                    param_mean = numpy.nanmean([c[key] for c in ecodes_params])
                    setattr(self.ecode, key, param_mean)

        else:
            logger.warning(
                "Could not compute average ecode for protocol {} target {} "
                "because it didn't match any recordings".format(
                    self.name, self.amplitude
                )
            )

    def get_files_used(self):

        files = []

        for rec in self.recordings:
            files += rec.files

        return files

    def identify(self, name, target):
        return name == self.name and target == self.amplitude

    def mean_stimulus(self):

        if len(self.recordings):
            return self.ecode.generate()
        else:
            logger.warning(
                "Cannot generate time and current series for protocol {} "
                "because its target didn't match any recordings".format(
                    self.name
                )
            )
            return ([], [])

    def ecode_params(self):

        out = {
            "holding": {
                "delay": 0.0,
                "amp": self.ecode.hypamp,
                "duration": self.ecode.tend,
                "totduration": self.ecode.tend,
            },
            "step": self.ecode.get_stimulus_parameters(),
        }

        return out
