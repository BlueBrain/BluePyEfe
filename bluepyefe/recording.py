"""Recording class"""

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
import efel
import matplotlib.pyplot as plt

from .tools import to_ms, to_mV, to_nA, set_efel_settings

logger = logging.getLogger(__name__)


class Recording(object):

    """Contains the data related to an electrophysiological recording."""

    def __init__(self, config_data, reader_data, protocol_name):
        """
        Constructor

        Args:
            config_data (dict): metadata for the recording considered informed
                by the user.
            reader_data (dict): metadata for the recording considered returned
                by the recording reader.
            protocol_name (str): name of the protocol of the present
                recording.
        """

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

        self.export_attr = None

    @property
    def time(self):
        """Alias of the time attribute"""
        return self.t

    @time.setter
    def time(self, value):
        """Setter for an alias of the time attribute"""
        self.t = value

    def set_timing_ecode(self, name_timings, config_data):
        """Used by some of the children classes to check that the timing of
        the ecode is provided by the user and assign it to the correct
        attribute."""

        for timing in name_timings:
            if timing in config_data and config_data[timing] is not None:
                setattr(self, timing, int(round(config_data[timing] / self.dt)))
            else:
                raise AttributeError(
                    f"For protocol {self.protocol_name}, {timing} should"
                    "be specified in the config (in ms)."
                )

    def set_amplitudes_ecode(self, amp_name, config_data, reader_data, value):
        """Check in the user-provided data or reader-provided data if a
         given current amplitude is provided. If it isn't use the value
         computed from the current series"""

        if amp_name in config_data and config_data[amp_name] is not None:
            setattr(self, amp_name, config_data[amp_name])
        elif amp_name in reader_data and reader_data[amp_name] is not None:
            setattr(self, amp_name, reader_data[amp_name])
        else:
            setattr(self, amp_name, value)

    def timing_index_to_ms(self, name_timing, time_series):
        """Used by some of the children classes to translate a timing attribute
         from index to ms."""

        setattr(self, name_timing, time_series[int(round(getattr(self, name_timing)))])

    def get_params(self):
        """Returns the eCode parameters"""
        return {attr: getattr(self, attr) for attr in self.export_attr}

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
        # WARNING: the ljp is informed as a positive float but we substract it
        # from the voltage
        if "ljp" in config_data and config_data["ljp"] is not None:
            voltage = voltage - config_data["ljp"]

        if "repetition" in reader_data:
            self.repetition = reader_data["repetition"]

        return t, current, voltage

    def call_efel(self, efeatures, efel_settings=None):
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

        try:
            return efel.getFeatureValues(
                [efel_trace], efeatures, raise_warnings=False
            )
        except TypeError as e:
            if "Unknown feature name" in str(e):
                str_f = " ".join(efeatures)
                raise Exception("One of the following feature name does not "
                                f"exist in eFEL: {str_f}")

    def compute_efeatures(
        self, efeatures, efeature_names=None, efel_settings=None
    ):
        """Compute a set of efeatures for the present recording.

        Args:
            efeatures (list of str): name of the efeatures to extract from
                the recordings.
            efeature_names (list of str): Optional. Given name for the
                features. Can and should be used if the same feature
                is to be extracted several time on different sections
                of the same recording.
            efel_settings (dict): eFEL settings in the form
                {setting_name: setting_value}.
        """

        if efeature_names is None:
            efeature_names = efeatures

        for i, f in enumerate(efeatures):
            if efeature_names[i] is None:
                efeature_names[i] = f

        efel_vals = self.call_efel(efeatures, efel_settings)
        for efeature_name, efeature in zip(efeature_names, efeatures):

            value = efel_vals[0][efeature]
            if value is None or numpy.isinf(numpy.nanmean(value)):
                value = numpy.nan

            self.efeatures[efeature_name] = numpy.nanmean(value)

    def compute_spikecount(self, efel_settings=None):
        """Compute the number of spikes in the trace"""
        if not efel_settings:
            efel_settings = {}

        tmp_settings = {'strict_stiminterval': True}
        tmp_settings.update(efel_settings)
        efel_vals = self.call_efel(['peak_time'], tmp_settings)

        self.spikecount = len(efel_vals[0]['peak_time'])

    def get_plot_amplitude_title(self):
        return " ({:.01f}%)".format(self.amp_rel)

    def plot(
        self,
        axis_current=None,
        axis_voltage=None,
        display_xlabel=True,
        display_ylabel=True
    ):
        """Plot the recording"""

        if axis_current is None or axis_voltage is None:
            _, axs = plt.subplots(nrows=2, ncols=1, figsize=[4.9, 4.8])
            axis_current, axis_voltage = axs[0], axs[1]

        title = "Amp = {:.03f} nA".format(self.amp)
        if self.amp_rel is not None:
            title += self.get_plot_amplitude_title()
        if self.id is not None:
            title += "\nid: {}".format(self.id)
        if self.repetition is not None:
            title += "\nRepetition: {}".format(self.repetition)
        axis_current.set_title(title, size="x-small")

        gen_t, gen_i = self.generate()
        axis_current.plot(self.t, self.current, c="C0")
        axis_current.plot(gen_t, gen_i, c="C1", ls="--")
        axis_voltage.plot(self.t, self.voltage, c="C0")

        if display_xlabel:
            axis_voltage.set_xlabel("Time (ms)")
        if display_ylabel:
            axis_current.set_ylabel("Current (nA)")
            axis_voltage.set_ylabel("Voltage (mV)")

        axis_current.tick_params(axis="both", which="major", labelsize=8)
        axis_current.tick_params(axis="both", which="minor", labelsize=6)
        axis_voltage.tick_params(axis="both", which="major", labelsize=8)
        axis_voltage.tick_params(axis="both", which="minor", labelsize=6)

        return axis_current, axis_voltage
