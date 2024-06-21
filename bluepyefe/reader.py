"""Trace reader functions"""

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
import h5py
import numpy
import scipy.io
from neo import io
import os

from . import igorpy
from .nwbreader import BBPNWBReader, ScalaNWBReader, AIBSNWBReader

logger = logging.getLogger(__name__)


def _check_metadata(metadata, reader_name, required_entries=[]):

    for entry in required_entries:

        if entry not in metadata:

            raise KeyError(
                "The trace reader {} expects the metadata {}. The "
                "entry {} was not provided.".format(
                    reader_name, ", ".join(e for e in required_entries), entry
                )
            )


def axon_reader(in_data):
    """Reader to read .abf

    Args:
        in_data (dict): of the format

            .. code-block:: python

                {
                    "filepath": "./XXX.abf",
                    "i_unit": "pA",
                    "t_unit": "s",
                    "v_unit": "mV"
                }
    """

    fp = in_data["filepath"]
    r = io.AxonIO(filename=fp)
    bl = r.read_block(lazy=False)

    data = []
    for trace in bl.segments:

        dt = 1.0 / int(trace.analogsignals[0].sampling_rate)
        np_trace = numpy.asarray(trace.analogsignals)

        if np_trace.shape[0] == 2:
            v = np_trace[0, :]
            c = np_trace[1, :]
        elif np_trace.shape[-1] == 2:
            v = np_trace[:, :, 0]
            c = np_trace[:, :, 1]
        else:
            raise Exception(f"Unknown .abf format for file {fp}. Maybe "
                            "it does not have current data?")

        data.append({
            "voltage": numpy.asarray(v).flatten(),
            "current": numpy.asarray(c).flatten(),
            "dt": dt
        })

    return data


def igor_reader(in_data):
    """Reader to read old .ibw

    Args:
        in_data (dict): of the format

            .. code-block:: python

                {
                    'i_file': './XXX.ibw',
                    'v_file': './XXX.ibw',
                    'v_unit': 'V',
                    't_unit': 's',
                    'i_unit': 'A'
                }
    """

    _check_metadata(
        in_data, igor_reader.__name__, ["v_file", "i_file", "t_unit"]
    )

    # Read file
    notes_v, voltage = igorpy.read(in_data["v_file"])
    notes_i, current = igorpy.read(in_data["i_file"])

    if "A" in notes_v.dUnits and "V" in notes_i.dUnits:

        logger.warning(
            "It seems that the i_file and v_file are reversed for file: "
            "{}".format(in_data["v_file"])
        )

        voltage, current = current, voltage
        notes_v, notes_i = notes_i, notes_v

    # Extract data
    trace_data = {}
    trace_data["voltage"] = numpy.asarray(voltage)
    trace_data["v_unit"] = str(notes_v.dUnits).replace(" ", "")
    trace_data["dt"] = notes_v.dx
    trace_data["current"] = numpy.asarray(current)
    trace_data["i_unit"] = str(notes_i.dUnits).replace(" ", "")

    return [trace_data]


def read_matlab(in_data):
    """To read .mat from http://gigadb.org/dataset/100535

    Args:
        in_data (dict): of the format

            .. code-block:: python

                {
                    'filepath': './161214_AL_113_CC.mat',
                    'ton': 2000,
                    'toff': 2500,
                    'v_unit': 'V',
                    't_unit': 's',
                    'i_unit': 'A'
                }
    """

    _check_metadata(
        in_data,
        read_matlab.__name__,
        ["filepath", "i_unit", "v_unit", "t_unit"],
    )

    r = scipy.io.loadmat(in_data["filepath"])

    data = []
    for k, v in r.items():

        if "Trace" in k and k[-1] == "1":

            trace_data = {
                "current": v[:, 1],
                "voltage": r[k[:-1] + "2"][:, 1],
                "dt": v[1, 0],
            }

            data.append(trace_data)

    return data


def nwb_reader(in_data):
    """Reader for .nwb

    Args:
        in_data (dict): of the format

            .. code-block:: python

                {
                    'filepath': './XXX.nwb',
                    "protocol_name": "IV",
                    "repetition": 1 (or [1, 3, ...]) # Optional
                }
    """

    _check_metadata(
        in_data,
        nwb_reader.__name__,
        ["filepath", "protocol_name"],
    )

    target_protocols = in_data['protocol_name']
    if isinstance(target_protocols, str):
        target_protocols = [target_protocols]

    with h5py.File(in_data["filepath"], "r") as content:
        if "data_organization" in content:
            reader = BBPNWBReader(
                content,
                target_protocols,
                in_data.get("repetition", None),
                in_data.get("v_file", None)
            )
        elif "timeseries" in content["acquisition"].keys():
            reader = AIBSNWBReader(content, target_protocols)
        else:
            reader = ScalaNWBReader(content, target_protocols)

        data = reader.read()

    return data


def csv_lccr_reader(in_data):
    """Reader to read .txt (csv_lccr)

    Args:
        in_data (dict): of the format:

            .. code-block:: python

                {
                    'filepath': "./XXX.txt",
                    'dt': 0.1,
                    'ton': 2000,
                    'toff': 2500,
                    'ljp': 14.0,
                    'amplitudes': [10 -10 20 -20 30 -30 40 -40 50 -50],
                    'hypamp': -20 # (units should match 'amplitudes'),
                    'remove_last_100ms': True,
                    'v_unit': 'mV',
                    't_unit': 'ms',
                    'i_unit': 'pA' # current unit for 'amplitudes' and 'hypamp'
                }
    """
    _check_metadata(
        in_data,
        csv_lccr_reader.__name__,
        ["filepath", "dt", "amplitudes", "v_unit", "t_unit", "i_unit", "ton", "toff", "hypamp"],
    )

    data = []

    fln = os.path.join(in_data['filepath'])
    if not os.path.isfile(fln):
        raise FileNotFoundError(
            "Please provide a string with the filename of the txt file; "
            f"current path not found: {fln}"
        )

    dt = in_data['dt']
    ton = in_data['ton']
    toff = in_data['toff']
    amplitudes = in_data['amplitudes']
    hypamp = in_data['hypamp']

    import csv
    with open(fln, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        columns = list(zip(*reader))
        length = numpy.shape(columns)[1]

        voltages = numpy.array([
            [
                float(string) if string not in ["-", ""] else 0
                for string in column
            ]
            for column in columns
        ])
        t = numpy.arange(length) * dt

    # Remove last 100 ms if needed
    if in_data.get('remove_last_100ms', False):
        slice_end = int(-100. / dt)
        voltages = voltages[:, :slice_end]
        t = t[:slice_end]

    for amplitude, voltage in zip(amplitudes, voltages):
        current = numpy.zeros_like(voltage)
        ion, ioff = int(ton / dt), int(toff / dt)
        current[:] = hypamp
        current[ion:ioff] = amplitude + hypamp
        trace_data = {
            "filename": os.path.basename(in_data['filepath']),
            "current": current,
            "voltage": voltage,
            "t": t,
            "dt": numpy.float64(dt),
            "ton": numpy.float64(ton),
            "toff": numpy.float64(toff),
            "amp": numpy.float64(amplitude),
            "hypamp": numpy.float64(hypamp),
            "ljp": in_data.get('ljp', 0),
            "i_unit": in_data['i_unit'],
            "v_unit": in_data['v_unit'],
            "t_unit": in_data['t_unit'],
        }

        data.append(trace_data)

    return data
