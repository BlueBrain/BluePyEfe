"""Igor reader"""

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

import numpy
from . import igorpy
from collections import OrderedDict

import logging

logger = logging.getLogger(__name__)


def process(config=None,
            filename=None,
            cellname=None,
            expname=None,
            stim_feats=None,
            idx_file=None,
            ljp=0, v_corr=0):
    """Read recordings from an igor file"""

    path = config['path']
    cells = config['cells']
    features = config['features']
    options = config['options']

    data = OrderedDict()
    data['voltage'] = []
    data['current'] = []
    data['dt'] = []

    data['t'] = []
    data['ton'] = []
    data['toff'] = []
    data['tend'] = []
    data['amp'] = []
    data['hypamp'] = []
    data['filename'] = []

    ordinal = filename['ordinal']
    logger.debug(" Adding igor file with ordinal %s", ordinal)

    i_file = path + filename['i_file']
    v_file = path + filename['v_file']

    notes, wave = igorpy.read(v_file)

    if 'v_unit' not in filename:
        v_unit = notes.xUnits
    else:
        v_unit = filename['v_unit']

    if v_unit == 'V':
        v = wave * 1e3  # mV
    elif v_unit == 'mV':
        v = wave  # mV
    else:
        raise Exception(
            "Unit voltage not configured!")

    if 'dt' in filename:
        dt = filename['dt']
        if numpy.isclose(dt, notes.dx) is False:
            raise Exception(
                "Given stepsize %f does not match stepsize from "
                "wavenotes %f" % (dt, notes.dt))
    else:
        dt = notes.dx

    if 't_unit' not in filename:
        t_unit = notes.dUnits
        filename['t_unit'] = t_unit
    else:
        t_unit = filename['t_unit']

    if (t_unit == ""):
        filename['t_unit'] = "s"

    if (t_unit == "") or (t_unit == "s"):
        dt = dt * 1e3  # convert to ms
    t = dt * numpy.arange(0, len(wave))

    notes, wave = igorpy.read(i_file)

    if 'i_unit' not in filename:
        i_unit = notes.xUnits
    else:
        i_unit = filename['i_unit']

    if i_unit == 'A':
        i = wave * 1e9  # nA
    elif i_unit == 'pA':
        i = wave * 1e-3  # nA
    else:
        raise Exception(
            "Unit current not configured!")

    ton = options['onoff'][expname][0]
    toff = options['onoff'][expname][1]

    ion = int(ton / dt)

    if toff:
        ioff = int(toff / dt)
    else:
        ioff = False

    hypamp = numpy.mean(i[0:ion])  # estimate hyperpolarization current
    iborder = int((ioff - ion) * 0.1)  # 10% distance to measure step current
    # depolarization amplitude starting from hypamp!!

    if expname in ['APThreshold']:
        imax = numpy.argmax(i)
        toff = t[imax]
        trun = toff - ton
        ampoff = numpy.mean(i[int(imax - 10. / dt):imax]) - hypamp
        # extrapolate to get expected amplitude at 1 sec
        amp = ampoff / trun * 2000.
        # amp = ampoff
    elif expname in ['H40S8']:
        amp = numpy.mean(i[i > 0.1])
    elif "pulseAmp" in filename:
        if i_unit.lower() == "a":
            amp = filename["pulseAmp"] * 1e9
        elif i_unit.lower() == "pa":
            amp = filename["pulseAmp"] * 1e-3
        else:
            amp = filename["pulseAmp"]
    else:
        amp = numpy.mean(i[ion + iborder:ioff - iborder]) - hypamp

    # clean voltage from transients
    if expname in ['IDRest', 'IDrest', 'IDthresh', 'IDdepol']:
        cut_start = int(ion + numpy.ceil(1.0 / dt))
        v[ion:cut_start] = v[cut_start]
        cut_end0 = int(ioff - numpy.ceil(0.5 / dt))
        cut_end1 = int(ioff + numpy.ceil(2.0 / dt))
        v[cut_end0:cut_end1] = v[cut_end1]

    # delete second pulse
    elif expname in ['SpikeRec']:
        t = t[:int(50. / dt)]
        v = v[:int(50. / dt)]
        i = i[:int(50. / dt)]

    # normalize membrane potential to known value
    if v_corr:
        v = v - numpy.mean(v[0:ion]) + v_corr

    v = v - ljp  # correct junction potential

    data['voltage'].append(v)
    data['current'].append(i)
    data['dt'].append(dt)

    data['t'].append(t)
    data['tend'].append(t[-1])
    data['ton'].append(ton)
    data['toff'].append(toff)
    data['amp'].append(amp)
    data['hypamp'].append(hypamp)
    data['filename'].append(ordinal)

    logger.debug(" Added igor file with ordinal %s", ordinal)

    return data
