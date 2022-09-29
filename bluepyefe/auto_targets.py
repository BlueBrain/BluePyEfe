"""class AutoTarget"""

"""
Copyright (c) 2021, EPFL/Blue Brain Project

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


class AutoTarget():

    """Fuzzy targets defined by lists of ecodes, amplitudes and efeatures.
    The AutoTarget will try to find the best combination of ecodes, amplitudes
     and efeatures that match the data available in the recordings."""

    def __init__(
        self,
        protocols,
        amplitudes,
        efeatures,
        min_recordings_per_amplitude=10,
        preferred_number_protocols=1,
        tolerance=10,
    ):
        """
        Extract efeatures.

        Args:
            protocols (list of str): list of eCodes, by order of priority.
            amplitudes (list of int): list of amplitudes, expressed as
                percentages of the rheobase, by order of priority
            efeatures (list str): list of efeatures.
            min_recordings_per_amplitude (int): minimum number of recordings
                a amplitude should have to be considered as a target.
            preferred_number_protocols (int): out of all the available
                amplitudes, how many should be used.
            tolerance (float): tolerance used to bin the efeatures around the
                target amplitudes.
        """

        self.efeatures = efeatures
        self.protocols = protocols
        self.amplitudes = amplitudes

        self.min_recordings_per_amplitude = min_recordings_per_amplitude
        self.preferred_number_protocols = preferred_number_protocols
        self.tolerance = tolerance

        self.active_amplitudes = []
        self.active_ecode = []

    def select_ecode_and_amplitude(self, recordings):
        """Based on what ephys data is available, builds an effective (active)
         set of targets."""

        for ecode in self.protocols:

            available_amps = []
            for r in recordings:
                if r.protocol_name == ecode and r.amp_rel is not None:
                    available_amps.append(r.amp_rel)

            for amp in self.amplitudes:
                n_available_rec = sum(
                    abs(a - amp) < self.tolerance for a in available_amps
                )
                if n_available_rec >= self.min_recordings_per_amplitude:
                    self.active_ecode.append(ecode)
                    self.active_amplitudes.append(amp)

        if (
            self.active_ecode and
            len(self.active_ecode) > self.preferred_number_protocols
        ):
            self.active_ecode = self.active_ecode[
                :self.preferred_number_protocols]
            self.active_amplitudes = self.active_amplitudes[
                :self.preferred_number_protocols]

    def is_valid(self):
        """Check if the present AutoTarget has active targets (if matching
         ephys data were found)"""

        return bool(self.active_amplitudes) and bool(self.active_ecode)

    def generate_targets(self):
        """Build a list of targets in the format expected by the main
        extraction function of BluePyEfe using the targets presently
        active."""

        targets = []

        for amp, protocol_name in zip(
                self.active_amplitudes, self.active_ecode):
            for efeature in self.efeatures:
                targets.append({
                    "efeature": efeature,
                    "protocol": protocol_name,
                    "amplitude": amp,
                    "tolerance": self.tolerance,
                })

        return targets


def default_auto_targets():
    """Define a set of 3 generic AutoTarget for firing pattern properties,
     AP waveform properties and hyperpolarizing step properties."""

    auto_firing_pattern = AutoTarget(
        protocols=["Step", "FirePattern", "IDrest", "IDRest", "IDthresh",
                   "IDThresh", "IDThres", "IDthres", "IV"],
        amplitudes=[200, 150, 250, 300],
        efeatures=['voltage_base',
                   'adaptation_index2',
                   'mean_frequency',
                   'time_to_first_spike',
                   'time_to_last_spike',
                   'inv_first_ISI',
                   'inv_second_ISI',
                   'inv_third_ISI',
                   'inv_fourth_ISI',
                   'inv_fifth_ISI',
                   'inv_last_ISI',
                   'ISI_CV',
                   'ISI_log_slope',
                   'doublet_ISI',
                   'AP_amplitude',
                   'AP1_amp',
                   'APlast_amp',
                   'AHP_depth',
                   'AHP_time_from_peak'],
        min_recordings_per_amplitude=1,
        preferred_number_protocols=2,
        tolerance=25.,
    )

    auto_ap_waveform = AutoTarget(
        protocols=["APWaveform", "APwaveform", "Step", "FirePattern",
                   "IDrest", "IDRest", "IDthresh", "IDThresh", "IDThres",
                   "IDthres", "IV"],
        amplitudes=[300, 350, 250, 400, 200],
        efeatures=["AP_amplitude",
                   "AP1_amp",
                   "AP2_amp",
                   'AP_width',
                   "AP_duration_half_width",
                   "AP_rise_time",
                   "AP_fall_time",
                   'AHP_depth_abs',
                   'AHP_time_from_peak',
                   "AHP_depth"],
        min_recordings_per_amplitude=1,
        preferred_number_protocols=1,
        tolerance=25.,
    )

    auto_iv = AutoTarget(
        protocols=["IV", "Step"],
        amplitudes=[-50, -100],
        efeatures=['voltage_base',
                   'steady_state_voltage_stimend',
                   'ohmic_input_resistance_vb_ssse',
                   'voltage_deflection',
                   'voltage_deflection_begin',
                   'decay_time_constant_after_stim',
                   'Spikecount',
                   'sag_ratio1',
                   'sag_ratio2',
                   'sag_amplitude',
                   'sag_time_constant'],
        min_recordings_per_amplitude=1,
        preferred_number_protocols=1,
        tolerance=10.,
    )

    return [auto_firing_pattern, auto_ap_waveform, auto_iv]
