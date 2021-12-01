"""Tool functions"""

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
import json
import numpy
import efel


DEFAULT_EFEL_SETTINGS = {
    'strict_stiminterval': True,
    'Threshold': -20.,
    'interp_step': 0.025
}


def to_ms(t, t_unit):
    if t_unit == "s" or t_unit == "sec":
        return t * 1e3
    elif t_unit == "ms":
        return t
    elif t_unit == "10th_ms":
        return t * 0.1
    else:
        raise Exception("Time unit '{}' is unknown.".format(t_unit))


def to_nA(current, i_unit):
    if i_unit == "A":
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
    if v_unit == "V":
        return voltage * 1e3
    elif v_unit == "uV":
        return voltage * 1e-3
    elif v_unit == "mV":
        return voltage
    else:
        raise Exception("Voltage unit '{}' is unknown.".format(v_unit))


def merge_efel_settings(efeature_settings):
    """Combine the current efeature's settings with the default ones"""

    return {**DEFAULT_EFEL_SETTINGS, **efeature_settings}


def set_efel_settings(efeature_settings):
    """ Reset the eFEl settings and set them as requested by the user (uses
        default value otherwise).
    """

    efel.reset()

    settings = merge_efel_settings(efeature_settings)

    for setting, value in settings.items():

        if setting == 'Threshold':
            efel.setThreshold(value)

        elif isinstance(value, bool) or isinstance(value, int):
            efel.setIntSetting(setting, int(value))

        elif isinstance(value, float):
            efel.setDoubleSetting(setting, value)

        elif isinstance(value, str):
            efel.setStrSetting(setting, value)


def dict_to_json(data, path):
    """
    Save some data in a json file.
    """
    s = json.dumps(data, indent=2, cls=NumpyEncoder)
    with open(path, "w") as f:
        f.write(s)


class NumpyEncoder(json.JSONEncoder):
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
            obj, (numpy.float_, numpy.float16, numpy.float32, numpy.float64)
        ):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
