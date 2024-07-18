"""EFeatureTarget class"""

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
import numpy
import logging

logger = logging.getLogger(__name__)


class EFeatureTarget():

    """E-feature target defined by an efeature to compute for a given protocol
    and amplitude. Contains the resulting values"""

    def __init__(
            self,
            efeature_name,
            efel_feature_name,
            protocol_name,
            amplitude,
            tolerance,
            efel_settings=None,
    ):
        """Constructor.

        Args:
            efeature_name (str): name of the feature (can be different than
                the efel_feature_name - e.g. Spikecount_phase1)
            efel_feature_name (str): name of the eFeature in the eFEL library
                (ex: 'AP1_peak')
            protocol_name (str): name of the recording from which the efeature
                should be computed
            amplitude (float): amplitude of the current stimuli for which the
                efeature should be computed (expressed as a percentage of the
                threshold amplitude (rheobase))
            tolerance (float): tolerance around the target amplitude in which
                an experimental recording will be seen as a hit during
                efeatures extraction (expressed as a percentage of the
                threshold amplitude (rheobase))
            efel_settings (dict): target specific efel settings.
        """

        self.efeature_name = efeature_name
        if self.efeature_name is None:
            self.efeature_name = efel_feature_name
        self.efel_feature_name = efel_feature_name
        self.protocol_name = protocol_name

        self.amplitude = amplitude
        self.tolerance = tolerance

        self.efel_settings = efel_settings
        if self.efel_settings is None:
            self.efel_settings = {}

        self._values = []
        self._files = []
        self._auto_thresholds = []

    @property
    def values(self):
        """Return all values."""
        return self._values

    @property
    def mean(self):
        """Average of the e-feature value at target"""

        return numpy.nanmean(self._values)

    @property
    def std(self):
        """Standard deviation of the e-feature value at target"""

        return numpy.nanstd(self._values)

    @property
    def sample_size(self):
        """Number of value that matched the target present"""

        return len(self._values)

    def append(self, value, files=None):
        """Append a feature value to the present target"""

        if not isinstance(value, (int, float)):
            raise TypeError("Expected value of type int or float")

        if numpy.isnan(value) or value is None:
            logger.info(
                "Trying to append {} to efeature {} for protocol {} {}. Value "
                "will be ignored".format(
                    value,
                    self.efel_feature_name,
                    self.protocol_name,
                    self.amplitude
                )
            )
            return

        self._values.append(value)
        if files:
            self._files += files

    def clear(self):
        """Clear the list of feature values"""

        self._values = []
        self._files = []

    def add_effective_threshold(self):
        """If auto threshold detection was used during feature extraction,
        update the efel settings with the Threshold that was actually used"""

        if self._auto_thresholds:
            self.efel_settings["Threshold"] = numpy.median(self._auto_thresholds)

    def as_dict(self, save_files_used=False, default_std_value=1e-3):
        """Returns the target in the form of a dictionary in a legacy format"""

        self.add_effective_threshold()

        std = self.std
        if std == 0.0:
            logger.warning(
                "Standard deviation for efeatures {} stimulus {} is 0 and "
                "will be set to {}".format(
                    self.efel_feature_name, self.protocol_name, default_std_value
                )
            )
            std = default_std_value

        feature_dict = {
            "feature": self.efel_feature_name,
            "val": [self.mean, std],
            "n": self.sample_size,
            "efel_settings": self.efel_settings
        }

        if self.efeature_name:
            feature_dict['efeature_name'] = self.efeature_name
        if save_files_used:
            feature_dict['files'] = self.files

        return feature_dict

    def __str__(self):
        """String representation"""

        str_form = "Target E-Feature {} for protocol {} {:.1f}%:\n".format(
            self.efel_feature_name, self.protocol_name, self.amplitude
        )

        str_form += "Sample size (n): {}".format(self.sample_size)

        if self.sample_size:
            str_form += "\nMean: {:.5f}\nStandard deviation: {:.5f}".format(
                self.mean, self.std
            )

        return str_form
