"""sAHP eCode"""

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
from .tools import scipy_signal2d
from .tools import base_current

logger = logging.getLogger(__name__)


class SAHP(Recording):

    """sAHP current stimulus

    The long step (here amp) is usually fixed at 40% of rheobase, and the short step (here amp2)
    can usually vary from 150% to 300% of rheobase.

    .. code-block:: none

           hypamp        hypamp+amp       hypamp+amp2        hypamp+amp           hypamp
             :                :                :                 :                   :
             :                :          ______________          :                   :
             :                :         |              |         :                   :
             :                :         |              |         :                   :
             :      ____________________                ____________________         :
             :     |                    ^              ^                    |        :
             :     |                    :              :                    |        :
        |__________|                    :              :                    |__________________
        ^          ^                    :              :                    ^                  ^
        :          :                    :              :                    :                  :
        :          :                    :              :                    :                  :
        t=0        ton                  tmid           tmid2                toff               tend
    """

    def __init__(
        self,
        config_data,
        reader_data,
        protocol_name="sAHP",
        efel_settings=None
    ):

        super(SAHP, self).__init__(config_data, reader_data, protocol_name)

        self.ton = None
        self.tmid = None
        self.tmid2 = None
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

        if self.voltage is not None:
            self.set_autothreshold()
            self.compute_spikecount(efel_settings)

        self.export_attr = ["ton", "tmid", "tmid2", "toff", "tend", "amp",
                            "amp2", "hypamp", "dt", "amp_rel", "amp2_rel",
                            "hypamp_rel"]

    def get_stimulus_parameters(self):
        """Returns the eCode parameters"""
        ecode_params = {
            "delay": self.ton,
            "tmid": self.tmid,
            "tmid2": self.tmid2,
            "toff": self.toff,
            "amp": self.amp2,
            "long_amp": self.amp,
            "thresh_perc": self.amp2_rel,
            "duration": self.tmid2 - self.tmid,
            "totduration": self.tend,
        }
        return ecode_params

    def compute_amp(self, current, config_data, reader_data):

        smooth_current = scipy_signal2d(current, 85)

        hypamp_value = numpy.median(
            numpy.concatenate(
                (smooth_current[: self.ton], smooth_current[self.toff :])
            )
        )
        self.set_amplitudes_ecode("hypamp", config_data, reader_data, hypamp_value)

        amp_value = numpy.median(
            numpy.concatenate(
                (
                    smooth_current[self.ton: self.tmid],
                    smooth_current[self.tmid2: self.toff],
                )
            )
        ) - self.hypamp
        self.set_amplitudes_ecode("amp", config_data, reader_data, amp_value)

        amp2_value = numpy.median(smooth_current[self.tmid : self.tmid2]) - self.hypamp
        self.set_amplitudes_ecode("amp2", config_data, reader_data, amp2_value)

        if config_data.get("tend", None) is None:
            self.tend = len(self.t) * self.dt
        else:
            self.tend = config_data["tend"]

        self.ton = self.t[int(round(self.ton))]
        self.toff = self.t[int(round(self.toff))]
        self.tmid = self.t[int(round(self.tmid))]
        self.tmid2 = self.t[int(round(self.tmid2))]

    def step_detection(self, current, config_data, reader_data):

        # Set the threshold to detect the step
        noise_level = numpy.std(numpy.concatenate((self.current[:50], self.current[-50:])))
        step_threshold = numpy.max([2.0 * noise_level, 1e-5])

        # The buffer prevent miss-detection of the step when artifacts are
        # present at the very start or very end of the current trace
        buffer_detect = 2.0
        idx_buffer = int(buffer_detect / self.dt)
        idx_buffer = max(1, idx_buffer)

        buffer_step = 50
        smooth_current = scipy_signal2d(current, 85)

        # Infer the beginning of the long step
        self.hypamp = base_current(current)

        if "ton" in config_data and config_data["ton"] is not None:
            self.ton = int(round(config_data["ton"] / self.dt))
        else:
            tmp_current = numpy.abs(smooth_current[idx_buffer:] - self.hypamp)
            self.ton = idx_buffer + numpy.argmax(tmp_current > step_threshold)

        # Infer the end of the long step
        tmp_current = numpy.flip(
            numpy.abs(smooth_current[self.ton:-idx_buffer] - self.hypamp)
        )
        self.toff = (
            (len(current) - numpy.argmax(tmp_current > step_threshold)) - 1 - idx_buffer
        )

        # Get the amplitude of the step current (relative to hypamp)
        self.amp = numpy.median(
            numpy.concatenate((smooth_current[self.ton:self.ton + 50],
                               smooth_current[self.toff - 50:self.toff]))
        ) - self.hypamp

        # Infer the beginning of the short step
        tmp_current = numpy.abs(
            smooth_current[self.ton + buffer_step:self.toff - buffer_step] -
            self.amp - self.hypamp
        )
        self.tmid = self.ton + buffer_step + numpy.argmax(tmp_current > step_threshold)

        # Infer the end of the long step
        tmp_current = numpy.flip(
            numpy.abs(
                smooth_current[self.ton + buffer_step:self.toff - 50] -
                self.amp - self.hypamp
            )
        )
        self.tmid2 = (
            (self.toff - numpy.argmax(tmp_current > step_threshold)) - 1 - buffer_step
        )

        self.amp2 = numpy.median(smooth_current[self.tmid:self.tmid2]) - self.hypamp

        # Converting back ton and toff to ms
        self.ton = self.t[int(round(self.ton))]
        self.toff = self.t[int(round(self.toff))]
        self.tmid = self.t[int(round(self.tmid))]
        self.tmid2 = self.t[int(round(self.tmid2))]
        self.tend = len(self.t) * self.dt

        # Check for some common step detection failures when the current
        # is constant.
        if self.ton >= self.toff or self.ton >= self.tend or \
                self.toff > self.tend or self.tmid == self.ton \
                or self.tmid2 == self.toff:

            self.ton = 0.
            self.toff = self.tend

            logger.warning(
                "The automatic step detection failed for the recording "
                f"{self.protocol_name} in files {self.files}. You should "
                "specify ton and toff by hand in your files_metadata "
                "for this file."
            )

    def interpret(self, t, current, config_data, reader_data):
        """Analyse a current with a step and extract from it the parameters
        needed to reconstruct the array"""

        self.dt = t[1]
        required = ["ton", "tmid", "tmid2", "toff"]
        if all(r in config_data for r in required):
            self.set_timing_ecode(required, config_data)
            self.compute_amp(current, config_data, reader_data)
        else:
            self.step_detection(current, config_data, reader_data)

    def generate(self):
        """Generate the current array from the parameters of the ecode"""

        ton = int(self.ton / self.dt)
        tmid = int(self.tmid / self.dt)
        tmid2 = int(self.tmid2 / self.dt)
        toff = int(self.toff / self.dt)

        time = numpy.arange(0.0, self.tend, self.dt)
        current = numpy.full(time.shape, numpy.float64(self.hypamp))
        current[ton:tmid] += numpy.float64(self.amp)
        current[tmid2:toff] += numpy.float64(self.amp)
        current[tmid:tmid2] += numpy.float64(self.amp2)

        return time, current

    def compute_relative_amp(self, amp_threshold):
        self.amp_rel = 100.0 * self.amp / amp_threshold
        self.amp2_rel = 100.0 * self.amp2 / amp_threshold
        self.hypamp_rel = 100.0 * self.hypamp / amp_threshold

    def in_target(self, target, tolerance, absolute_amplitude=False):
        """Returns a boolean. True if the amplitude of the eCode is close to
        target and False otherwise."""

        effective_amp = self.amp2 if absolute_amplitude else self.amp2_rel

        if numpy.abs(target - effective_amp) < tolerance:
            return True

        return False

    def get_plot_amplitude_title(self):
        return " ({:.01f}%/{:.01f}%)".format(self.amp_rel, self.amp2_rel)
