"""Recording class"""

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
import efel

from .tools import to_ms, to_mV, to_nA, set_efel_settings

logger = logging.getLogger(__name__)


class Recording(object):

    def __init__(self, config_data, reader_data, protocol_name):

        self.config_data = config_data
        self.reader_data = reader_data
        self.protocol_name = protocol_name

        if "filepath" in config_data:
            self.files = [config_data["filepath"]]
        elif "i_file" in config_data:
            self.files = [config_data["i_file"], config_data["v_file"]]
        else:
            self.files = []

        self.id = reader_data.get("id", None)

        self.location = None
        self.efeatures = {}
        self.spikecount = None

        self.t = None
        self.current = None
        self.voltage = None

        self.repetition = None

        if len(reader_data):
            self.t, self.current, self.voltage = self.standardize_trace(
                config_data, reader_data
            )

    def get_params(self):
        """Returns the eCode parameters"""
        params = {}
        return params

    def interpret(self):
        """Analyse a current array and extract from it the parameters needed to
        reconstruct the array"""
        pass

    def generate(self):
        """Generate the current array from the parameters of the ecode"""
        pass

    def compute_relative_amp(self, amp_threshold):
        """Divide all the amplitude in the stimuli by the spiking amplitude"""
        pass

    def in_target(self, target, tolerance):
        """Returns a boolean. True if the amplitude of the eCode is close to
        target and False otherwise."""
        pass

    def standardize_trace(self, config_data, reader_data):
        """Standardize the units of the current and voltage times series. If
        some metadata are present both in the file itself and the file_metadata
         dictionary, the latter is used."""

        # Create the time series
        t = numpy.arange(len(reader_data["voltage"]))

        # Give it the correct sampling rate
        if "dt" in config_data and config_data["dt"] is not None:
            t = t * config_data["dt"]
        elif "dt" in reader_data and reader_data["dt"] is not None:
            t = t * reader_data["dt"]
        else:
            raise Exception(
                "Sampling rate not configured for "
                "file {}".format(self.files)
            )

        # Convert it to ms
        if "t_unit" in config_data and config_data["t_unit"] is not None:
            t = to_ms(t, config_data["t_unit"])
        elif "t_unit" in reader_data and reader_data["t_unit"] is not None:
            t = to_ms(t, reader_data["t_unit"])
        else:
            raise Exception(
                "Time unit not configured for " "file {}".format(self.files)
            )

        # Convert current to nA
        if "i_unit" in config_data and config_data["i_unit"] is not None:
            current = to_nA(reader_data["current"], config_data["i_unit"])
        elif "i_unit" in reader_data and reader_data["i_unit"] is not None:
            current = to_nA(reader_data["current"], reader_data["i_unit"])
        else:
            raise Exception(
                "Current unit not configured for " "file {}".format(self.files)
            )

        # Convert voltage to mV
        if "v_unit" in config_data and config_data["v_unit"] is not None:
            voltage = to_mV(reader_data["voltage"], config_data["v_unit"])
        elif "v_unit" in reader_data and reader_data["v_unit"] is not None:
            voltage = to_mV(reader_data["voltage"], reader_data["v_unit"])
        else:
            raise Exception(
                "Voltage unit not configured for " "file {}".format(self.files)
            )

        # Offset membrane potential to known value
        if "v_corr" in config_data and config_data["v_corr"] is not None:
            voltage = (
                voltage - numpy.median(voltage[:100]) + config_data["v_corr"]
            )

        # Correct for the liquid junction potential
        # WARNING: the ljp is past as a positive float but we substract it from the voltage
        if "ljp" in config_data and config_data["ljp"] is not None:
            voltage = voltage - config_data["ljp"]

        if "repetition" in reader_data:
            self.repetition = reader_data["repetition"]

        return t, current, voltage

    def compute_spikecount(self):
        """Compute the number of spikecounts in the trace"""

        efel_trace = {
            "T": self.t,
            "V": self.voltage,
            "stim_start": [self.ton],
            "stim_end": [self.toff]
        }

        set_efel_settings({"stimulus_current": self.amp})

        efel_vals = efel.getFeatureValues(
            [efel_trace], ['peak_time'], raise_warnings=False
        )

        self.spikecount = len(efel_vals[0]['peak_time'])

    def compute_efeatures(self, efeatures, efel_settings=None):
        """ Calls efel to computed the wanted efeature """

        if efel_settings is None:
            efel_settings = {}

        settings = {"stimulus_current": self.amp}
        for setting in efel_settings:
            if setting not in ['stim_start', 'stim_end']:
                settings[setting] = efel_settings[setting]
        set_efel_settings(settings)

        efel_trace = {
            "T": self.t,
            "V": self.voltage,
            'stim_start': [efel_settings.get('stim_start', self.ton)],
            'stim_end': [efel_settings.get('stim_end', self.toff)]
        }

        efel_vals = efel.getFeatureValues(
            [efel_trace], efeatures, raise_warnings=False
        )

        for efeature in efeatures:

            value = efel_vals[0][efeature]
            if value is None or numpy.isinf(numpy.nanmean(value)):
                value = numpy.nan

            self.efeatures[efeature] = numpy.nanmean(value)
