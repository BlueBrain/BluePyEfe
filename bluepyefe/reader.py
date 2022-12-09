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
            "filepath": "./XXX.abf",
            "i_unit": "pA",
            "t_unit": "s",
            "v_unit": "mV",
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


def _format_nwb_trace(voltage, current, start_time, trace_name=None, repetition=None):

    v_array = numpy.array(
        voltage[()] * voltage.attrs["conversion"], dtype="float32"
    )

    i_array = numpy.array(
        current[()] * current.attrs["conversion"], dtype="float32"
    )

    dt = 1. / float(start_time.attrs["rate"])

    v_unit = voltage.attrs["unit"]
    i_unit = current.attrs["unit"]
    t_unit = start_time.attrs["unit"]
    if not isinstance(v_unit, str):
        v_unit = voltage.attrs["unit"].decode('UTF-8')
        i_unit = current.attrs["unit"].decode('UTF-8')
        t_unit = start_time.attrs["unit"].decode('UTF-8')

    return {
        "voltage": v_array,
        "current": i_array,
        "dt": dt,
        "id": str(trace_name),
        "repetition": repetition,
        "i_unit": i_unit,
        "v_unit": v_unit,
        "t_unit": t_unit,
    }


def _nwb_reader_AIBS(content, target_protocols):

    data = []

    for sweep in list(content["acquisition"]["timeseries"].keys()):

        protocol_name = content["acquisition"]["timeseries"][sweep]["aibs_stimulus_name"][()]
        if not isinstance(protocol_name, str):
            protocol_name = protocol_name.decode('UTF-8')

        if target_protocols and protocol_name not in target_protocols:
            continue

        data.append(_format_nwb_trace(
            voltage=content["acquisition"]["timeseries"][sweep]["data"],
            current=content["stimulus"]["presentation"][sweep]["data"],
            start_time=content["acquisition"]["timeseries"][sweep]["starting_time"],
            trace_name=sweep
        ))

    return data


def _nwb_reader_Scala(content, target_protocols):

    data = []

    for sweep in list(content['acquisition'].keys()):

        key_current = sweep.replace('Series', 'StimulusSeries')
        protocol_name = "Step"

        if target_protocols and protocol_name not in target_protocols:
            continue

        if key_current not in content['stimulus']['presentation']:
            continue

        data.append(_format_nwb_trace(
            voltage=content['acquisition'][sweep]['data'],
            current=content['stimulus']['presentation'][key_current]['data'],
            start_time=content["acquisition"][sweep]["starting_time"],
            trace_name=sweep,
        ))

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


def _nwb_reader_BBP(content, target_protocols, repetition):

    data = []

    for ecode in target_protocols:

        for cell_id in content["data_organization"].keys():

            if ecode not in content["data_organization"][cell_id]:
                logger.debug(f"No eCode {ecode} in nwb.")
                continue

            ecode_content = content["data_organization"][cell_id][ecode]

            rep_iter = _get_repetition_keys_nwb(
                ecode_content, request_repetitions=repetition
            )

            for rep in rep_iter:

                for sweep in ecode_content[rep].keys():
                    for trace_name in list(ecode_content[rep][sweep].keys()):

                        if "ccs_" in trace_name:
                            key_current = trace_name.replace("ccs_", "ccss_")
                        elif "ic_" in trace_name:
                            key_current = trace_name.replace("ic_", "ics_")
                        else:
                            continue

                        if key_current not in content["stimulus"]["presentation"]:
                            logger.debug(f"Ignoring {key_current} not"
                                         " present in the stimulus presentation")
                            continue
                        if trace_name not in content["acquisition"]:
                            logger.debug(f"Ignoring {trace_name} not"
                                         " present in the acquisition")
                            continue

                        data.append(_format_nwb_trace(
                            voltage=content["acquisition"][trace_name]["data"],
                            current=content["stimulus"]["presentation"][key_current]["data"],
                            start_time=content["stimulus"]["presentation"][key_current]["starting_time"],
                            trace_name=trace_name,
                            repetition=int(rep.replace("repetition ", ""))
                        ))

    return data


def nwb_reader(in_data):
    """ Reader for .nwb

    Args:
        in_data (dict): of the format
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
            return _nwb_reader_BBP(content, target_protocols, in_data.get("repetition", None))
        elif "timeseries" in content["acquisition"].keys():
            return _nwb_reader_AIBS(content, target_protocols)
        else:
            return _nwb_reader_Scala(content, target_protocols)
