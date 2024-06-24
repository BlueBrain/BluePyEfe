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


def group_indexes(values, gap=10):
    """Return a list of clusters from a list where consecutive
    values follow each other forming clusters
    eg: [12, 14, 15, 20, 56, 60, 61, 62, 63] -> [[12, 14, 15, 20], [56, 60,
    61, 62, 63]]"""

    clusters = []

    for v in values:
        if not (len(clusters)) or clusters[-1][-1] + gap < v:
            clusters.append([v])
        else:
            clusters[-1].append(v)

    return clusters


def detect_spike(amp, hypamp, smooth_current, dt):

    tspike = []
    duration = []
    delta = []

    threshold = hypamp + (0.1 * amp)
    candidate_spikes = numpy.argwhere(smooth_current > threshold).flatten()
    candidate_spikes = group_indexes(candidate_spikes, gap=10)

    for spike in candidate_spikes:
        tspike.append(spike[0] - 1)
        duration.append(spike[-1] - spike[0] + 1)

    if len(tspike) > 1:
        for i in range(1, len(tspike)):
            end = tspike[i - 1] + duration[i - 1]
            start = tspike[i]
            delta.append(dt * (start - end))

    tspike = numpy.asarray(tspike) * dt
    duration = numpy.mean(numpy.asarray(duration) * dt)
    delta = numpy.mean(delta)

    return tspike, duration, delta


class SpikeRec(Recording):

    """SpikeRec current stimulus

    .. code-block:: none

              hypamp        hypamp+amp      hypamp       hypamp+amp          .   .   .
                :                :             :             :
                :        _________________     :      _________________                      _________________
                :       |                 |    :     |                 |                    |                 |
                :       |                 |    :     |                 |   * len(tspike)    |                 |
                :       |                 |    :     |                 |     .   .   .      |                 |
                :       |                 |    :     |                 |                    |                 |
        |_______________|                 |__________|                 |__                __|                 |___
        :               :                 :          :                 :                                          ^
        :               :                 :          :                 :                                          :
        :               :                 :          :                 :                                          :
         <--tspike[0] --><-spike_duration-><- delta -><-spike_duration->      .   .   .                          tend

    """

    def __init__(
        self,
        config_data,
        reader_data,
        protocol_name="SpikeRec",
        efel_settings=None
    ):

        super(SpikeRec, self).__init__(config_data, reader_data, protocol_name)

        self.tend = None
        self.tspike = []
        self.spike_duration = None  # in ms
        self.amp = None
        self.hypamp = None
        self.dt = None
        self.delta = None  # Time difference between two spikes

        self.amp_rel = None
        self.hypamp_rel = None

        if self.t is not None and self.current is not None:
            self.interpret(
                self.t, self.current, self.config_data, self.reader_data
            )

        if self.voltage is not None:
            self.set_autothreshold()
            self.compute_spikecount(efel_settings)

        self.export_attr = ["tend", "tspike", "spike_duration", "delta",
                            "amp", "hypamp", "dt", "amp_rel", "hypamp_rel"]

    @property
    def ton(self):
        return 0.0

    @property
    def toff(self):
        return self.tend

    @property
    def multi_stim_start(self):
        return list(self.tspike)

    @property
    def multi_stim_end(self):
        return [t + self.spike_duration for t in self.tspike]

    def get_stimulus_parameters(self):
        """Returns the eCode parameters"""

        ecode_params = {
            "delay": self.tspike[0],
            "n_spikes": len(self.tspike),
            "delta": self.delta,
            "amp": self.amp,
            "thresh_perc": self.amp_rel,
            "spike_duration": self.spike_duration,
            "totduration": self.tend,
        }

        return ecode_params

    def interpret(self, t, current, config_data, reader_data):
        """Analyse a current with a step and extract from it the parameters
        needed to reconstruct the array"""
        self.dt = t[1]

        # Smooth the current
        smooth_current = scipy_signal2d(current, 15)

        hypamp_value = base_current(current)
        self.set_amplitudes_ecode("hypamp", config_data, reader_data, hypamp_value)

        amp_value = numpy.max(smooth_current)
        self.set_amplitudes_ecode("amp", config_data, reader_data, amp_value)

        # Get the beginning and end of the spikes
        if (
            not len(self.tspike)
            or self.spike_duration is None
            or self.delta is None
        ):
            self.tspike, self.spike_duration, self.delta = detect_spike(
                self.amp, self.hypamp, smooth_current, self.dt
            )

        self.tend = len(t) * self.dt

    def generate(self):
        """Generate the step current array from the parameters of the ecode"""

        t = numpy.arange(0.0, self.tend, self.dt)
        current = numpy.full(t.shape, numpy.float64(self.hypamp))

        spike_start = int(self.tspike[0] / self.dt)
        spike_end = int((self.tspike[0] + self.spike_duration) / self.dt)
        current[spike_start:spike_end] += numpy.float64(self.amp)

        for i in range(1, len(self.tspike)):
            spike_start = int(spike_end + (self.delta / self.dt))
            spike_end = spike_start + int(self.spike_duration / self.dt)
            current[spike_start:spike_end] += numpy.float64(self.amp)

        return t, current

    def in_target(self, target, tolerance, absolute_amplitude):
        """Returns a boolean. True if the delta of the eCode is close to
        target and False otherwise."""
        logger.warning(
            "The eCode SpikeRec uses delta between current spikes "
            "in ms as target, not amplitude"
        )
        if numpy.abs(target - self.delta) < tolerance:
            return True
        else:
            return False
