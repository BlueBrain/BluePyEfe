"""Axon reader"""

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

from neo import io
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
import numpy
import os
import pprint
import neo.rawio.axonrawio


def process(config=None,
            filename=None,
            cellname=None,
            expname=None,
            stim_feats=None,
            idx_file=None,
            ljp=0, v_corr=0):
    """Read recordings from an axon file"""
    path = config['path']
    cells = config['cells']
    # features = config['features']
    # options = config['options']

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

    logger.debug(" Adding axon file %s", filename)

    f = os.path.join(path, cellname, filename + '.abf')
    r = io.AxonIO(filename=f)

    # read header

    # Below line doesn't work anymore due to api change
    # Now using rawio
    # header = r.read_header()

    header = neo.rawio.axonrawio.parse_axon_soup(f)

    # read sampling rate
    sampling_rate = 1.e6 / header['protocol']['fADCSequenceInterval']

    dt = 1. / int(sampling_rate) * 1e3
    # version = header['fFileVersionNumber']  # read file version
    bl = r.read_block(lazy=False)

    stim_info = None
    if 'stim_info' in cells[cellname]['experiments'][expname]:
        stim_info = cells[cellname]['experiments'][expname]['stim_info']

    else:

        # read stimulus features if present
        stim_feats = []
        if 'stim_feats' in cells[cellname]['experiments'][expname]:
            stim_feats = cells[cellname]['experiments'][expname]['stim_feats']

        all_stims = []
        if stim_feats:
            res = stim_feats_from_meta(stim_feats, len(bl.segments), idx_file)
            if res[0]:
                all_stims = res[1]
            else:
                print(res[1])
        if not all_stims:
            res = stim_feats_from_header(header)
            if res[0]:
                all_stims = res[1]
            else:
                pprint.pprint(
                    "No valid stimulus was found in metadata or files. \
                        Skipping current file")
                return

    # for all segments in file
    for i_seg, seg in enumerate(bl.segments):

        # dt = 1./int(seg.analogsignals[0].sampling_rate) * 1e3

        if stim_info is not None:

            voltage = numpy.array(
                seg.analogsignals[0]).astype(
                numpy.float64).flatten()
            current = numpy.array(
                seg.analogsignals[1]).astype(
                numpy.float64).flatten()
            t = numpy.arange(len(voltage)) * dt

            ton = stim_info['ton']
            toff = stim_info['toff']
            ion = int(ton / dt)
            ioff = int(toff / dt)

            if 'tamp' in stim_info:
                tamp = [int(stim_info['tamp'][0] / dt),
                        int(stim_info['tamp'][1] / dt)]
            else:
                tamp = [ion, ioff]

            i_unit = stim_info['i_unit']

            if i_unit == 'A':
                current = current * 1e9  # nA
            elif i_unit == 'pA':
                current = current * 1e-3  # nA
            else:
                raise Exception(
                    "Unit current not configured!")

            amp = numpy.nanmean(current[tamp[0]:tamp[1]])
            hypamp = numpy.nanmean(current[0:ion])

        else:

            voltage = numpy.array(seg.analogsignals[0]).astype(numpy.float64)
            t = numpy.arange(len(voltage)) * dt

            ton = all_stims[i_seg][1]
            toff = all_stims[i_seg][2]
            amp = numpy.float64(all_stims[i_seg][3])

            ion = int(ton / dt)
            ioff = int(toff / dt)

            current = []
            current = numpy.zeros(len(voltage))
            current[ion:ioff] = amp

            # estimate hyperpolarization current
            hypamp = numpy.mean(current[0:ion])

            # clean voltage from transients
            voltage[ion:ion + int(numpy.ceil(0.4 / dt))] = \
                voltage[ion + int(numpy.ceil(0.4 / dt))]
            voltage[ioff:ioff + int(numpy.ceil(0.4 / dt))] = \
                voltage[ioff + int(numpy.ceil(0.4 / dt))]

        # normalize membrane potential to known value (given in UCL excel
        # sheet)
        if v_corr:
            if len(v_corr) == 1 and v_corr[0] != 0.0:
                voltage = voltage - numpy.mean(voltage[0:ion]) + v_corr[0]
            elif len(v_corr) - 1 >= idx_file and v_corr[idx_file] != 0.0:
                voltage = voltage - numpy.mean(voltage[0:ion]) \
                    + v_corr[idx_file]

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


def stim_feats_from_meta(stim_feats, num_segments, idx_file):
    """Get stimulus features from metadata (author: Luca Leonardo
        Bologna)"""
    if not stim_feats:
        return (0, "Empty metadata in file")
    elif len(stim_feats) - 1 < idx_file and len(stim_feats) != 1:
        return (0, "Stimulus dictionaries are different \
                from the number of files")
    else:
        # array for storing all stimulus features
        all_stim_feats = []

        # for every segment in the axon file
        for i in range(num_segments):

            # read current stimulus dict
            if len(stim_feats) == 1:
                crr_dict = stim_feats[0]
            else:
                crr_dict = stim_feats[idx_file]

            # read stimulus information
            ty = str(crr_dict['stimulus_type'])
            tu = crr_dict['stimulus_time_unit']
            st = crr_dict['stimulus_start']
            en = crr_dict['stimulus_end']
            u = str(crr_dict['stimulus_unit'])
            fa = float(format(crr_dict['stimulus_first_amplitude'], '.3f'))
            inc = float(format(crr_dict['stimulus_increment'], '.3f'))
            if tu == 's':
                st = st * 1e3
                en = en * 1e3
            # compute current stimulus amplitude
            crr_val = float(format(fa + inc * float(format(i, '.3f')), '.3f'))
            crr_stim_feats = (ty, st, en, crr_val, u)

            # store current tuple
            all_stim_feats.append(crr_stim_feats)
        return (1, all_stim_feats)


def stim_feats_from_header(header):
    """Get stimulus features from the file header (author: Luca Leonardo
    Bologna)"""
    # read sampling rate
    sampling_rate = 1.e6 / header['protocol']['fADCSequenceInterval']

    # read file version
    version = header['fFileVersionNumber']

    # extract protocol for version >=.2
    if version >= 2.:
        # prot = r.read_protocol() # read protocol
        # read info for DAC
        dictEpochInfoPerDAC = header['dictEpochInfoPerDAC']

        # if field is empty
        if not (dictEpochInfoPerDAC):
            return (0, "No 'dictEpochInfoPerDAC' field")

        # if field is not empty, read all stimulus segments
        else:
            valid_epoch_dicts = [k for k, v in dictEpochInfoPerDAC.items()
                                 if bool(v)]

            # if more than one channel is activated for the stimulus
            # or a number of epochs different than 3 is found
            if len(valid_epoch_dicts) != 1 or len(dictEpochInfoPerDAC[0]) != 3:
                return (0, 'Exiting. More than one channel used \
                        for stimulation')
            else:
                # read all stimulus epochs

                # TODO IS THIS CORRECT ?
                k = valid_epoch_dicts[-1]
                stim_epochs = dictEpochInfoPerDAC[k]
                # read enabled waveforms
                stim_ch_info = [(i['DACChNames'], i['DACChUnits'],
                                 i['nDACNum']) for i in header['listDACInfo']
                                if bool(i['nWaveformEnable'])]

                # if epoch initial levels and increment are not
                # compatible with a step stimulus
                if (stim_epochs[0]['fEpochInitLevel'] !=
                        stim_epochs[2]['fEpochInitLevel'] or
                    stim_epochs[0]['fEpochLevelInc'] !=
                    stim_epochs[2]['fEpochLevelInc'] or
                    float(format(stim_epochs[0]['fEpochLevelInc'], '.3f')) != 0
                    or (len(stim_ch_info) != 1 or
                        stim_ch_info[0][2] != k)):
                    # return 0 with message
                    return (0, "A stimulus different from the steps \
                                has been detected")
                else:
                    ty = "step"
                    u = stim_ch_info[0][1]
                    # number of ADC channels
                    nADC = header['sections']['ADCSection']['llNumEntries']
                    # number of samples per episode
                    nSam = header['protocol']['lNumSamplesPerEpisode'] / nADC
                    # number of actual episodes
                    nEpi = header['lActualEpisodes']

                    # read first stimulus epoch
                    e_zero = header['dictEpochInfoPerDAC'][
                        stim_ch_info[0][2]][0]
                    # read second stimulus epoch
                    e_one = header['dictEpochInfoPerDAC'][
                        stim_ch_info[0][2]][1]
                    # index of stimulus beginning
                    i_last = int(nSam * 15625 / 10**6)
                    # create array for all stimulus info
                    all_stim_info = []

                    # step increment
                    e_one_inc = float(format(e_one['fEpochLevelInc'],
                                             '.3f'))
                    # step initial level
                    e_one_init_level = float(format(e_one['fEpochInitLevel'],
                                                    '.3f'))

                    # for every episode, compute stimulus start, stimulus end,
                    # stimulus value
                    for epiNum in range(nEpi):
                        st = i_last + e_zero['lEpochInitDuration'] + \
                            e_zero['lEpochDurationInc'] * epiNum
                        en = st + e_one['lEpochInitDuration'] +  \
                            e_one['lEpochDurationInc'] * epiNum
                        crr_val_full = float(format(e_one_init_level +
                                                    e_one_inc * epiNum, '.3f'))
                        crr_val = float(format(crr_val_full, '.3f'))
                        st = 1 / sampling_rate * st * 1e3
                        en = 1 / sampling_rate * en * 1e3
                        all_stim_info.append((ty, st, en, crr_val, u))
                    return (1, all_stim_info)
