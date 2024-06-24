"""Step eCode class"""

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
import logging
import numpy

from ..recording import Recording
from .tools import base_current
from .tools import scipy_signal2d

logger = logging.getLogger(__name__)


class Ramp(Recording):

    """Ramp current stimulus

    .. code-block:: none

            hypamp        hypamp+amp      hypamp
              :                :             :
              :                :             :
              :               /|             :
              :              / |             :
              :             /  |             :
              :            /   |             :
              :           /    |             :
              :          /     |             :
              :         /      |             :
              :        /       |             :
              :       /        |             :
        |___________ /         |__________________________
        ^           ^          ^                          ^
        :           :          :                          :
        :           :          :                          :
        t=0         ton        toff                       tend
    """

    def __init__(
        self,
        config_data,
        reader_data,
        protocol_name="ramp",
        efel_settings=None
    ):

        super(Ramp, self).__init__(config_data, reader_data, protocol_name)

        self.ton = None
        self.toff = None
        self.tend = None
        self.amp = None
        self.hypamp = None
        self.dt = None

        self.amp_rel = None
        self.hypamp_rel = None

        if self.t is not None and self.current is not None:
            self.interpret(
                self.t, self.current, self.config_data, self.reader_data
            )

        if self.voltage is not None:
            self.set_autothreshold()
            self.compute_spikecount(efel_settings)

        self.export_attr = ["ton", "toff", "tend", "amp", "hypamp", "dt",
                            "amp_rel", "hypamp_rel"]

    def get_stimulus_parameters(self):
        """Returns the eCode parameters"""
        ecode_params = {
            "delay": self.ton,
            "amp": self.amp,
            "thresh_perc": self.amp_rel,
            "duration": self.toff - self.ton,
            "totduration": self.tend,
        }
        return ecode_params

    def interpret(self, t, current, config_data, reader_data):
        """Analyse a current with a step and extract from it the parameters
        needed to reconstruct the array"""
        self.dt = t[1]

        # Smooth the current
        smooth_current = scipy_signal2d(current, 85)

        self.set_timing_ecode(["ton"], config_data)

        if "toff" in config_data and config_data["toff"] is not None:
            self.toff = int(round(config_data["toff"] / self.dt))
        else:
            self.toff = numpy.argmax(smooth_current[self.ton:]) + self.ton

        hypamp_value = base_current(current)
        self.set_amplitudes_ecode("hypamp", config_data, reader_data, hypamp_value)

        amp_value = numpy.median(current[self.toff - 10 : self.toff]) - self.hypamp
        self.set_amplitudes_ecode("amp", config_data, reader_data, amp_value)

        # Converting back to ms
        for name_timing in ["ton", "toff"]:
            self.index_to_ms(name_timing, t)
        self.tend = len(t) * self.dt

    def generate(self):
        """Generate the step current array from the parameters of the ecode"""

        ton_idx = int(self.ton / self.dt)
        toff_idx = int(self.toff / self.dt)

        t = numpy.arange(0.0, self.tend, self.dt)
        current = numpy.full(t.shape, numpy.float64(self.hypamp))
        current[ton_idx:toff_idx] += numpy.linspace(
            0.0, self.amp, toff_idx - ton_idx
        )

        return t, current
