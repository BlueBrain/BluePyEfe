"""Cheops eCode"""

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

logger = logging.getLogger(__name__)


class NegCheops(Recording):

    # pylint: disable=line-too-long,anomalous-backslash-in-string

    """NegCheops current stimulus

    .. code-block:: none

           hypamp          hypamp+amp           hypamp            hypamp+amp           hypamp            hypamp+amp           hypamp
             :                  :                  :                   :                  :                   :                  :
             :                  :                  :                   :                  :                   :                  :
        |__________             :             ____________             :             ____________             :             ____________
        :          :\           :           /:            :\           :           /:            :\           :           /:            ^
        :          : \          :          / :            : \          :          / :            : \          :          / :            :
        :          :  \         :         /  :            :  \         :         /  :            :  \         :         /  :            :
        :          :   \        :        /   :            :   \        :        /   :            :   \        :        /   :            :
        :          :    \       :       /    :            :    \       :       /    :            :    \       :       /    :            :
        :          :     \      :      /     :            :     \      :      /     :            :     \      :      /     :            :
        :          :      \     :     /      :            :      \     :     /      :            :      \     :     /      :            :
        :          :       \    :    /       :            :       \    :    /       :            :       \    :    /       :            :
        :          :        \   :   /        :            :        \   :   /        :            :        \   :   /        :            :
        :          :         \  :  /         :            :         \  :  /         :            :         \  :  /         :            :
        :          :          \ : /          :            :          \ : /          :            :          \ : /          :            :
        :          :           \ /           :            :           \ /           :            :           \ /           :            :
        :          :            '            :            :            '            :            :            '            :            :
        :          :                         :            :                         :            :                         :            :
        t=0        ton                       t1           t2                        t3           t4                        toff         tend
    """

    def __init__(
        self,
        config_data,
        reader_data,
        protocol_name="NegCheops",
        efel_settings=None
    ):

        super(NegCheops, self).__init__(
            config_data, reader_data, protocol_name
        )

        self.ton = None
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None
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

        self.export_attr = ["ton", "t1", "t2", "t3", "t4", "toff", "tend",
                            "amp", "hypamp", "dt", "amp_rel", "hypamp_rel"]

    def get_stimulus_parameters(self):
        """Returns the eCode parameters"""
        ecode_params = {
            "delay": self.ton,
            "t1": self.t1,
            "t2": self.t2,
            "t3": self.t3,
            "t4": self.t4,
            "toff": self.toff,
            "amp": self.amp,
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

        self.set_timing_ecode(
            ["ton", "t1", "t2", "t3", "t4", "toff"], config_data)

        hypamp_value = numpy.median(smooth_current[: self.ton])
        self.set_amplitudes_ecode("hypamp", config_data, reader_data, hypamp_value)

        amp_value = numpy.min(smooth_current[:]) - self.hypamp
        self.set_amplitudes_ecode("amp", config_data, reader_data, amp_value)

        # Converting back to ms
        for name_timing in ["ton", "t1", "t2", "t3", "t4", "toff"]:
            self.index_to_ms(name_timing, t)
        self.tend = len(t) * self.dt

    def generate(self):
        """Generate the current array from the parameters of the ecode"""

        ton = int(self.ton / self.dt)
        t1 = int(self.t1 / self.dt)
        t2 = int(self.t2 / self.dt)
        t3 = int(self.t3 / self.dt)
        t4 = int(self.t4 / self.dt)
        toff = int(self.toff / self.dt)

        time = numpy.arange(0.0, self.tend, self.dt)
        current = numpy.full(time.shape, numpy.float64(self.hypamp))

        # First peak
        mid = int(0.5 * (ton + t1))
        current[ton:mid] += numpy.linspace(0.0, self.amp, mid - ton)
        current[mid:t1] += numpy.linspace(self.amp, 0.0, t1 - mid)

        # Second peak
        mid = int(0.5 * (t2 + t3))
        current[t2:mid] += numpy.linspace(0.0, self.amp, mid - t2)
        current[mid:t3] += numpy.linspace(self.amp, 0.0, t3 - mid)

        # Third peak
        mid = int(0.5 * (t4 + toff))
        current[t4:mid] += numpy.linspace(0.0, self.amp, mid - t4)
        current[mid:toff] += numpy.linspace(self.amp, 0.0, toff - mid)

        return time, current
