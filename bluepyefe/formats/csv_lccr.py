"""CSV lccr reader"""

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

import os

from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
import numpy


def process(config=None,
            filename=None,
            cellname=None,
            expname=None,
            stim_feats=None,
            idx_file=None,
            ljp=0, v_corr=0):
    """Read recordings from an csv file"""
    path = config['path']
    cells = config['cells']

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

    fln = os.path.join(path, cellname, filename + '.txt')
    if not os.path.exists(fln):
        raise Exception(
            'Please provide a string with filename of csv file, '
            'current path not found: %s',
            fln)

    exp_options = cells[cellname]['experiments'][expname]

    if (('dt' not in exp_options) or
        ('amplitudes' not in exp_options) or
        ('hypamp' not in exp_options) or
            ('ton' not in exp_options) or ('toff' not in exp_options)):
        raise Exception('Please provide additional options for LCCR csv')

    dt = exp_options['dt']
    hypamp = exp_options['hypamp']
    ton = exp_options['ton']
    toff = exp_options['toff']
    amplitudes = exp_options['amplitudes']

    import csv
    with open(fln, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        columns = list(zip(*reader))
        length = numpy.shape(columns)[1]

        for ic, column in enumerate(columns):

            voltage = numpy.zeros(length)
            for istr, string in enumerate(column):
                if (string != "-") and (string != ""):
                    voltage[istr] = float(string)

            t = numpy.arange(len(voltage)) * dt
            amp = amplitudes[ic]
            voltage = voltage - ljp  # correct liquid junction potential

            # remove last 100 ms
            voltage = voltage[0:int(-100. / dt)]
            t = t[0:int(-100. / dt)]
            ion = int(ton / dt)
            ioff = int(toff / dt)
            current = []
            current = numpy.zeros(len(voltage))
            current[ion:ioff] = amp

            if ('exclude' in cells[cellname] and
                any(abs(cells[cellname]['exclude'][idx_file] - amp) <
                    1e-4)):

                logger.info(" Not using trace with amplitude %f", amp)

            else:
                data['voltage'].append(voltage)
                data['current'].append(current)
                data['t'].append(t)

                data['dt'].append(numpy.float(dt))
                data['ton'].append(numpy.float(ton))
                data['toff'].append(numpy.float(toff))
                data['amp'].append(numpy.float(amp))
                data['hypamp'].append(numpy.float(hypamp))
                data['filename'].append(filename)

    return data
