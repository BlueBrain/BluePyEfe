"""ibf json reader"""

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
import os
import json

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
    """Read recordings from a json file"""

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

    # read stimulus features if present
    stim_feats = []
    if 'stim_feats' in cells[cellname]['experiments'][expname]:
        stim_feats = cells[cellname]['experiments'][expname]['stim_feats']

    logger.debug(" Adding ibf_json file %s", filename)

    f = path + os.sep + cellname + os.sep + filename + '.json'
    f = path + os.sep + filename + '.json'
    with open(f, 'r') as crr_file:
        f_data = json.load(crr_file)

    sampling_rate = f_data['sampling_rate']

    dt = 1. / int(sampling_rate) * 1e3

    # for all segments in file
    for idx, value in f_data['traces'].items():
        crr_amp = idx
        voltage = numpy.array(value).astype(numpy.float64)
        t = numpy.arange(len(voltage)) * dt

        ton = f_data['tonoff'][idx]['ton'][0]
        toff = f_data['tonoff'][idx]['toff'][0]
        ion = int(ton / dt)
        ioff = int(toff / dt)
        amp = numpy.float64(idx)

        current = []
        current = numpy.zeros(len(voltage))
        current[ion:ioff] = amp

        # estimate hyperpolarization current
        hypamp = numpy.mean(current[0:ion])

        # 10% distance to measure step current
        iborder = int((ioff - ion) * 0.1)

        # depolarization amplitude
        # amp = numpy.mean( current[ion+iborder:ioff-iborder] )
        voltage_dirty = voltage[:]

        # clean voltage from transients
        voltage[ion:ion + int(numpy.ceil(0.4 / dt))] = voltage[
            ion + int(numpy.ceil(0.4 / dt))]
        voltage[ioff:ioff + int(numpy.ceil(0.4 / dt))] = voltage[
            ioff + int(numpy.ceil(0.4 / dt))]

        # normalize membrane potential to known value (given in UCL
        # excel sheet)
        if v_corr:
            if len(v_corr) == 1 and v_corr[0] != 0.0:
                voltage = voltage - numpy.mean(voltage[0:ion]) + v_corr[0]
            elif len(v_corr) - 1 >= idx_file and v_corr[idx_file] != 0.0:
                voltage = voltage - numpy.mean(voltage[0:ion]) + v_corr[
                    idx_file]

        voltage = voltage - ljp

        # clip spikes after stimulus so they are not analysed
        voltage[ioff:] = numpy.clip(voltage[ioff:], -300, -40)

        if ('exclude' in cells[cellname] and
                any(abs(cells[cellname]['exclude'][idx_file] - amp) < 1e-4)):
            continue  # llb

        else:
            data['voltage'].append(voltage)
            data['current'].append(current)
            data['dt'].append(dt)

            data['t'].append(t)
            data['tend'].append(t[-1])
            data['ton'].append(ton)
            data['toff'].append(toff)
            data['amp'].append(amp)
            data['hypamp'].append(hypamp)
            data['filename'].append(filename)

    return data
