"""Trace reader functions"""

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
import h5py
import numpy
import scipy.io
from neo import io

from . import igorpy

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
        {
            "filepath": "./XXX.ibw",
            "i_unit": "pA",
            "t_unit": "s",
            "v_unit": "mV",
        }
    """

    _check_metadata(
        in_data,
        axon_reader.__name__,
        ["filepath", "i_unit", "v_unit", "t_unit"],
    )

    filepath = in_data["filepath"]

    # Read file
    r = io.AxonIO(filename=filepath)
    bl = r.read_block(lazy=False)

    # Extract data
    data = []
    for trace in bl.segments:
        trace_data = {}
        trace_data["voltage"] = numpy.array(trace.analogsignals[0]).flatten()
        trace_data["current"] = numpy.array(trace.analogsignals[1]).flatten()
        trace_data["dt"] = 1.0 / int(trace.analogsignals[0].sampling_rate)
        data.append(trace_data)

    return data


def igor_reader(in_data):
    """Reader to read old .ibw

    Args:
        in_data (dict): of the format
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
    trace_data["v_unit"] = notes_v.dUnits
    trace_data["dt"] = notes_v.dx
    trace_data["current"] = numpy.asarray(current)
    trace_data["i_unit"] = notes_i.dUnits

    return [trace_data]


def nwb_reader(in_data):
    """Reader to read old .nwb from LNMC

    Args:
        in_data (dict): of the format
        {
            'filepath': './XXX.nwb',
            'v_unit': 'V',
            't_unit': 's',
            'i_unit': 'A',
            "protocol_name": "Name_of_the_protocol"
        }
    """

    _check_metadata(
        in_data,
        nwb_reader.__name__,
        ["filepath", "i_unit", "v_unit", "t_unit", "protocol_name"],
    )

    filepath = in_data["filepath"]
    r = h5py.File(filepath, "r")

    data = []
    for sweep in list(r["acquisition"]["timeseries"].keys()):

        key_current = "Experiment_{}".format(sweep.replace("Sweep_", ""))
        protocol_name = str(
            r["acquisition"]["timeseries"][sweep]["aibs_stimulus_name"][()]
        )

        if protocol_name == in_data["protocol_name"]:

            trace_data = {
                "voltage": numpy.array(
                    r["acquisition"]["timeseries"][sweep]["data"][()],
                    dtype="float32",
                ),
                "current": numpy.array(
                    r["epochs"][key_current]["stimulus"]["timeseries"]["data"],
                    dtype="float32",
                ),
                "dt": 1.0
                / float(
                    r["acquisition"]["timeseries"][sweep][
                        "starting_time"
                    ].attrs["rate"]
                ),
                "id": str(key_current)
            }

            data.append(trace_data)

    return data


def _get_repetition_keys_nwb(ecode_content, request_repetitions=None):

    if isinstance(request_repetitions, (int, str)):
        request_repetitions = [int(request_repetitions)]

    reps = list(ecode_content.keys())
    reps_id = [int(rep.replace("repetition ", "")) for rep in reps]

    if request_repetitions:
        return [reps[reps_id.index(i)] for i in request_repetitions]
    else:
        return list(ecode_content.keys())


def _format_nwb_trace(content, trace_name, repetition):

    if "ccs_" in trace_name:
        key_current = trace_name.replace("ccs_", "ccss_")
    elif "ic_" in trace_name:
        key_current = trace_name.replace("ic_", "ics_")
    else:
        return None

    voltage = content["acquisition"][trace_name]["data"]
    v_array = numpy.array(
        voltage[()] * voltage.attrs["conversion"], dtype="float32"
    )

    current = content["stimulus"]["presentation"][key_current]["data"]
    i_array = numpy.array(
        current[()] * current.attrs["conversion"], dtype="float32"
    )

    time = content["stimulus"]["presentation"][key_current]["starting_time"]

    dt = 1.0 / float(time.attrs["rate"])
    v_unit = voltage.attrs["unit"]
    i_unit = current.attrs["unit"]
    t_unit = time.attrs["unit"]

    return {
        "voltage": v_array,
        "current": i_array,
        "dt": dt,
        "id": str(trace_name),
        "repetition": int(repetition.replace("repetition ", "")),
        "i_unit": i_unit,
        "v_unit": v_unit,
        "t_unit": t_unit,
    }


def nwb_reader_BBP(in_data):
    """ Reader to read .nwb from LNMC

    Args:
        in_data (dict): of the format
        {
            'filepath': './XXX.nwb',
            "protocol_name": "IV",
            "repetition": 1 (or [1, 3, ...])
        }
    """

    _check_metadata(
        in_data,
        nwb_reader_BBP.__name__,
        ["filepath", "protocol_name"],
    )

    filepath = in_data["filepath"]

    with h5py.File(filepath, "r") as content:

        data = []

        ecodes = in_data['protocol_name']
        if isinstance(ecodes, str):
            ecodes = [ecodes]

        for ecode in ecodes:

            for cell_id in content["data_organization"].keys():

                if ecode not in content["data_organization"][cell_id]:
                    logger.warning(
                        f"No eCode {ecode} in nwb {in_data['filepath']}."
                    )
                    continue

                ecode_content = content["data_organization"][cell_id][ecode]

                rep_iter = _get_repetition_keys_nwb(
                    ecode_content,
                    request_repetitions=in_data.get("repetition", None)
                )

                for rep in rep_iter:
                    for sweep in ecode_content[rep].keys():
                        for trace_name in list(ecode_content[rep][sweep].keys()):

                            formatted_trace = _format_nwb_trace(
                                content, trace_name, rep
                            )

                            if formatted_trace:
                                data.append(formatted_trace)

    return data


def read_matlab(in_data):
    """To read .mat from http://gigadb.org/dataset/100535

    Args:
        in_data (dict): of the format
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
