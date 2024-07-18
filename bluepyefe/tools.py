"""Tool and miscellaneous functions"""

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
import json
import numpy
import efel


DEFAULT_EFEL_SETTINGS = {
    'strict_stiminterval': True,
    'Threshold': -20.,
    'interp_step': 0.025
}


PRESET_PROTOCOLS_RHEOBASE = [
    "IV", "Step", "FirePattern", "IDrest", "IDRest", "IDthresh", "IDThresh", "IDThres", "IDthres"
]


def to_ms(t, t_unit):
    """Converts a time series to ms.

    Args:
        t (array): time series.
        t_unit (str): unit of the time series. Has to be "s", "sec",
            "seconds", "ms" or "10th_ms".
    """

    if t_unit.lower() in ["s", "sec", "seconds"]:
        return t * 1e3
    elif t_unit == "ms":
        return t
    elif t_unit == "10th_ms":
        return t * 0.1
    else:
        raise Exception("Time unit '{}' is unknown.".format(t_unit))


def to_nA(current, i_unit):
    """Converts a current series to nA.

    Args:
        current (array): current series.
        i_unit (str): unit of the current series. Has to be "a", "amperes",
            "amps", "mA", "uA", "pA" or "nA".
    """

    if i_unit.lower() in ["a", "amperes", "amps"]:
        return current * 1e9
    elif i_unit == "mA":
        return current * 1e6
    elif i_unit == "uA":
        return current * 1e3
    elif i_unit == "pA":
        return current * 1e-3
    elif i_unit == "nA":
        return current
    else:
        raise Exception("Current unit '{}' is unknown.".format(i_unit))


def to_mV(voltage, v_unit):
    """Converts a voltage series to mV.

    Args:
        voltage (array): voltage series.
        v_unit (str): unit of the voltage series. Has to be "v", "volts",
            "uV" or "mV".
    """

    if v_unit.lower() in ["v", "volts"]:
        return voltage * 1e3
    elif v_unit == "uV":
        return voltage * 1e-3
    elif v_unit == "mV":
        return voltage
    else:
        raise Exception("Voltage unit '{}' is unknown.".format(v_unit))


def set_efel_settings(efeature_settings):
    """Reset the eFEl settings and set them as requested by the user (uses
        default value otherwise).

    Args:
         efeature_settings (dict): eFEL settings in the form
            {setting_name: setting_value}.
    """

    efel.reset()

    for setting, value in efeature_settings.items():

        if setting in ['stim_start', 'stim_end']:
            value = float(value)

        efel.set_setting(setting, value)


def dict_to_json(data, path):
    """Save some data in a json file."""

    s = json.dumps(data, indent=2, cls=NumpyEncoder)
    with open(path, "w") as f:
        f.write(s)


class NumpyEncoder(json.JSONEncoder):

    """To make Numpy arrays JSON serializable"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                numpy.int_,
                numpy.intc,
                numpy.intp,
                numpy.int8,
                numpy.int16,
                numpy.int32,
                numpy.int64,
                numpy.uint8,
                numpy.uint16,
                numpy.uint32,
                numpy.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(
            obj, (numpy.float16, numpy.float32, numpy.float64)
        ):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
