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


class Step(Recording):

    """Step current stimulus

    ``      hypamp             hypamp+amp               hypamp             ``
    ``        :                     :                      :               ``
    ``        :                     :                      :               ``
    ``        :           ______________________           :               ``
    ``        :          |                      |          :               ``
    ``        :          |                      |          :               ``
    ``        :          |                      |          :               ``
    ``        :          |                      |          :               ``
    ``|__________________|                      |______________________    ``
    ``^                  ^                      ^                      ^   ``
    ``:                  :                      :                      :   ``
    ``:                  :                      :                      :   ``
    ``t=0                ton                    toff                   tend``
    """

    def __init__(
        self,
        config_data,
        reader_data,
        protocol_name="step",
        efel_settings=None
    ):

        super(Step, self).__init__(config_data, reader_data, protocol_name)

        self.ton = None
        self.toff = None
        self.tend = None
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
        smooth_current = None

        # Set the threshold to detect the step
        noise_level = numpy.std(numpy.concatenate((self.current[:50], self.current[-50:])))
        step_threshold = numpy.max([4.5 * noise_level, 1e-5])

        # The buffer prevent miss-detection of the step when artifacts are
        # present at the very start or very end of the current trace
        buffer_detect = 2.0
        idx_buffer = int(buffer_detect / self.dt)
        idx_buffer = max(1, idx_buffer)

        if "ton" in config_data and config_data["ton"] is not None:
            self.ton = int(round(config_data["ton"] / self.dt))
        elif "ton" in reader_data and reader_data["ton"] is not None:
            self.ton = int(round(reader_data["ton"]))
        else:
            self.ton = None

        # toff (index, not ms)
        if "toff" in config_data and config_data["toff"] is not None:
            self.toff = int(round(config_data["toff"] / self.dt))
        elif "toff" in reader_data and reader_data["toff"] is not None:
            self.toff = int(round(reader_data["toff"]))
        else:
            self.toff = None

        # Infer the begin and end of the step current
        if self.ton is None:
            if self.hypamp is None:
                self.hypamp = base_current(current)
            if smooth_current is None:
                smooth_current = scipy_signal2d(current, 85)
            _ = numpy.abs(smooth_current[idx_buffer:] - self.hypamp)
            self.ton = idx_buffer + numpy.argmax(_ > step_threshold)
        elif self.hypamp is None:
            # Infer the base current hypamp
            self.hypamp = base_current(current, idx_ton=self.ton)

        if self.toff is None:
            if smooth_current is None:
                smooth_current = scipy_signal2d(current, 85)
            _ = numpy.flip(
                numpy.abs(smooth_current[:-idx_buffer] - self.hypamp)
            )
            self.toff = (
                (len(current) - numpy.argmax(_ > step_threshold)) - 1 - idx_buffer
            )

        # Get the amplitude of the step current (relative to hypamp)
        if self.amp is None:
            self.amp = (
                numpy.median(current[self.ton : self.toff]) - self.hypamp
            )

        # Converting back ton and toff to ms
        self.ton = t[int(round(self.ton))]
        self.toff = t[int(round(self.toff))]

        self.tend = len(t) * self.dt

        # Check for some common step detection failures when the current
        # is constant.
        if self.ton >= self.toff or self.ton >= self.tend or \
                self.toff > self.tend:

            self.ton = 0.
            self.toff = self.tend

            logger.warning(
                "The automatic step detection failed for the recording "
                f"{self.protocol_name} in files {self.files}. You should "
                "specify ton and toff by hand in your files_metadata "
                "for this file."
            )

    def generate(self):
        """Generate the step current array from the parameters of the ecode"""
        ton_idx = int(self.ton / self.dt)
        toff_idx = int(self.toff / self.dt)

        t = numpy.arange(0.0, self.tend, self.dt)
        current = numpy.full(t.shape, numpy.float64(self.hypamp))
        current[ton_idx:toff_idx] += numpy.float64(self.amp)

        return t, current
