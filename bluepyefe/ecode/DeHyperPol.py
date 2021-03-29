"""DeHyperPol eCode"""

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

from ..recording import Recording
from .tools import scipy_signal2d

logger = logging.getLogger(__name__)


class DeHyperPol(Recording):
    def __init__(self, config_data, reader_data, protocol_name="DeHyperPol"):

        super(DeHyperPol, self).__init__(
            config_data, reader_data, protocol_name
        )

        self.ton = None
        self.tmid = None
        self.toff = None
        self.tend = None
        self.amp = None
        self.amp2 = None
        self.hypamp = None
        self.dt = None

        self.amp_rel = None
        self.amp2_rel = None
        self.hypamp_rel = None

        if self.t is not None and self.current is not None:
            self.interpret(
                self.t, self.current, self.config_data, self.reader_data
            )

    def get_params(self):
        """Returns the eCode parameters"""
        ecode_params = {
            "ton": self.ton,
            "tmid": self.tmid,
            "toff": self.toff,
            "tend": self.tend,
            "amp": self.amp,
            "amp2": self.amp2,
            "hypamp": self.hypamp,
            "dt": self.dt,
            "amp_rel": self.amp_rel,
            "amp2_rel": self.amp2_rel,
            "hypamp_rel": self.hypamp_rel,
        }
        return ecode_params

    def get_stimulus_parameters(self):
        """Returns the eCode parameters"""
        ecode_params = {
            "delay": self.ton,
            "tmid": self.tmid,
            "amp": self.amp,
            "amp2": self.amp2,
            "thresh_perc": self.amp_rel,
            "duration": self.toff - self.ton,
            "totduration": self.tend,
        }
        return ecode_params

    def interpret(self, t, current, config_data, reader_data):
        """Analyse a current array and extract from it the parameters
        needed to reconstruct the array"""
        self.dt = t[1]

        # Smooth the current
        smooth_current = scipy_signal2d(current, 85)

        if "ton" in config_data and config_data["ton"] is not None:
            self.ton = int(round(config_data["ton"] / self.dt))
        else:
            raise AttributeError(
                "For protocol {}, ton should be specified "
                "in the config (in ms)".format(self.protocol_name)
            )

        if "tmid" in config_data and config_data["tmid"] is not None:
            self.tmid = int(round(config_data["tmid"] / self.dt))
        else:
            raise AttributeError(
                "For protocol {}, tmid should be specified "
                "in the config (in ms)".format(self.protocol_name)
            )

        if "toff" in config_data and config_data["toff"] is not None:
            self.toff = int(round(config_data["toff"] / self.dt))
        else:
            raise AttributeError(
                "For protocol {}, toff should be pecified "
                "in the config (in ms)".format(self.protocol_name)
            )

        # hypamp
        if "hypamp" in config_data and config_data["hypamp"] is not None:
            self.hypamp = config_data["hypamp"]
        elif "hypamp" in reader_data and reader_data["hypamp"] is not None:
            self.hypamp = reader_data["hypamp"]
        else:
            # Infer the base current hypamp
            self.hypamp = numpy.median(
                numpy.concatenate(
                    (smooth_current[: self.ton], smooth_current[self.toff :])
                )
            )

        # amp with respect to hypamp
        if "amp" in config_data and config_data["amp"] is not None:
            self.amp = config_data["amp"]
        elif "amp" in reader_data and reader_data["amp"] is not None:
            self.amp = reader_data["amp"]
        else:
            self.amp = numpy.median(smooth_current[self.ton : self.tmid])
            self.amp -= self.hypamp

        # amp2 with respect to hypamp
        if "amp2" in config_data and config_data["amp2"] is not None:
            self.amp2 = config_data["amp2"]
        elif "amp2" in reader_data and reader_data["amp2"] is not None:
            self.amp2 = reader_data["amp2"]
        else:
            self.amp2 = numpy.median(smooth_current[self.tmid : self.toff])
            self.amp2 -= self.hypamp

        # Converting back to ms
        self.ton = t[int(round(self.ton))]
        self.tmid = t[int(round(self.tmid))]
        self.toff = t[int(round(self.toff))]
        self.tend = len(t) * self.dt

    def generate(self):
        """Generate the current array from the parameters of the ecode"""

        ton = int(self.ton / self.dt)
        tmid = int(self.tmid / self.dt)
        toff = int(self.toff / self.dt)

        time = numpy.arange(0.0, self.tend, self.dt)
        current = numpy.full(time.shape, self.hypamp)
        current[ton:tmid] += self.amp
        current[tmid:toff] += self.amp2

        return time, current

    def compute_relative_amp(self, amp_threshold):
        self.amp_rel = 100.0 * self.amp / amp_threshold
        self.amp2_rel = 100.0 * self.amp2 / amp_threshold
        self.hypamp_rel = 100.0 * self.hypamp / amp_threshold

    def in_target(self, target, tolerance):
        """Returns a boolean. True if the amplitude of the eCode is close to
        target and False otherwise."""
        if numpy.abs(target - self.amp_rel) < tolerance:
            return True
        else:
            return False
