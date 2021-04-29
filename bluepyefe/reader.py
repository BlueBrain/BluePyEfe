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


def nwb_reader_BBP(in_data):
    """ Reader to read .nwb from LNMC

    Args:
        in_data (dict): of the format
        {
            'filepath': './XXX.nwb',
            'v_unit': 'V',
            't_unit': 's',
            'i_unit': 'A',
            "protocol_name": "IV",
            "repetition": 1 (or [1, 3, ...])
        }
    """

    _check_metadata(
        in_data,
        nwb_reader_BBP.__name__,
        ["filepath", "i_unit", "v_unit", "t_unit", "protocol_name"],
    )

    filepath = in_data["filepath"]
    r = h5py.File(filepath, "r")

    data = []

    ecode = in_data['protocol_name']

    for cell_id in r["data_organization"].keys():

        if ecode not in r["data_organization"][cell_id]:
            raise Exception(
                f"No eCode {ecode} in nwb  {in_data['filepath']}."
            )

        av_reps = list(r["data_organization"][cell_id][ecode].keys())
        av_reps_id = [int(rep.replace("repetition ", "")) for rep in av_reps]

        if "repetition" in in_data and in_data["repetition"]:
            if isinstance(in_data["repetition"], list):
                rep_iter = [av_reps[av_reps_id.index(i)]
                            for i in in_data["repetition"]]
            else:
                rep_iter = [av_reps[av_reps_id.index(in_data["repetition"])]]
        else:
            rep_iter = r["data_organization"][cell_id][ecode].keys()

        for rep in rep_iter:

            for sweep in r["data_organization"][cell_id][ecode][rep].keys():

                sweeps = r["data_organization"][cell_id][ecode][rep][sweep]

                for trace in list(sweeps.keys()):

                    if "ccs_" in trace:
                        key_current = trace.replace("ccs_", "ccss_")
                    else:
                        continue

                    v = r["acquisition"][trace]
                    i = r["stimulus"]["presentation"][key_current]

                    trace_data = {
                        "voltage": numpy.array(
                            v["data"][()] * v["data"].attrs["conversion"],
                            dtype="float32"
                        ),
                        "current": numpy.array(
                            i["data"][()] * i["data"].attrs["conversion"],
                            dtype="float32",
                        ),
                        "dt": 1.0 / float(v["starting_time"].attrs["rate"]),
                        "id": str(trace),
                        "repetition": int(rep.replace("repetition ", ""))
                    }

                    data.append(trace_data)

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
