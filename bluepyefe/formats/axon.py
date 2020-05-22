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
logger.setLevel(logging.INFO)
import numpy
import os
import sys
import neo.rawio.axonrawio
import json
from . import common


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

    # initialize data dictionary
    data = common.manageDicts.initialize_data_dict()

    # read metadata file if present
    meta_dict = {}

    f_meta = os.path.join(path, cellname, filename + '_metadata.json')
    if os.path.exists(f_meta):
        with open(f_meta) as json_metadata:
            meta_dict = json.load(json_metadata)
        json_metadata.close()

    logger.info(" Adding axon file %s", filename)

    f = os.path.join(path, cellname, filename + '.abf')
    r = io.AxonIO(filename=f)

    # read header

    # Below line doesn't work anymore due to api change
    # Now using rawio
    # header = r.read_header()

    header = neo.rawio.axonrawio.parse_axon_soup(f)

    # get number of episodes from header
    nbepisod = get_nbepisod(header)

    # read sampling rate
    # sampling_rate = 1.e6 / header['protocol']['fADCSequenceInterval']
    # dt = 1. / int(sampling_rate) * 1e3
    # version = header['fFileVersionNumber']

    # read data block
    bl = r.read_block(lazy=False)

    # initialize all stimuli data structure
    all_stims = []

    stim_feats = None
    res = []
    stim_info_flag = False

    if 'stim_info' in cells[cellname]['experiments'][expname] and \
            'stim_feats' in cells[cellname]['experiments'][expname]:
        raise Exception(
            "Both 'stim_info' and 'stim_feats' are given in" +
            " the configuration parameters. Only one or none is allowed.")

    # read stimulus info from the option stim_feats if any
    if 'stim_feats' in cells[cellname]['experiments'][expname]:
        stim_feats_crr = cells[cellname]['experiments'][expname]['stim_feats']

        # extract stimulus info from the config file
        if isinstance(stim_feats_crr, dict) is True:
            stim_feats = stim_feats_crr
        elif isinstance(stim_feats_crr, list) is True:
            if len(stim_feats_crr) == 1:
                stim_feats = stim_feats_crr[0]
            elif len(stim_feats_crr) == len(cells[cellname][
                    'experiments'][expname]['files']):
                stim_feats = stim_feats_crr[idx_file]
            else:
                raise ValueError(
                    "The 'stim_feats' list must have length " +
                    "equals to 1 or to the length of the 'files' list." +
                    "Please, check your configuration")
        else:
            raise ValueError(
                "'stim_feats' must be a list or a dictionary. Please," +
                " check your configuration")

        stim_feats['filename'] = filename
        res = common.manageMetadata.stim_feats_from_meta(
            stim_feats, nbepisod)

        logger.info("File: %s. Found info in config file", filename)

    # extract stim from metadata file if any
    elif 'stim_info' in cells[cellname]['experiments'][expname] and \
            cells[cellname]['experiments'][expname]['stim_info'] is not None:
        stim_info = cells[cellname]['experiments'][expname]['stim_info']
        stim_info_flag = True
        try:
            sampling_rate = 1.e6 / header['protocol']['fADCSequenceInterval']
        except Exception as e:
            logger.info(
                "Unable to find recording frequency in file: %s." +
                "The ABF file version is probably older than v2", filename)

    else:
        # read metadata file if present
        stim_feats = {}

        f_meta = os.path.join(path, cellname, filename + '_metadata.json')
        if os.path.exists(f_meta):
            with open(f_meta) as json_metadata:
                stim_feats = json.load(json_metadata)
            json_metadata.close()

        # if metadata with stimulus info could be extracted
        if stim_feats:
            res = common.manageMetadata.stim_feats_from_meta(
                meta_dict, nbepisod)
            logger.info("File: %s. Found info in metadata file", filename)

    # if stimulus has been extracted from stim_feats or metadata file
    if res and res[0]:
        # extract sampling rate from metadata
        sampling_rate = res[1]["r"][0]
        all_stims = res[1]
    elif not stim_info_flag:
        # extract stim from header if any
        logger.info(
            "File: %s. No stimulus info found in config or " +
            "metadata file. Extracting info from the file header.",
            filename)
        res = stim_feats_from_header(header)
        all_stims = res[1]

        # extract sampling rate from header
        sampling_rate = sampling_rate_from_header(header)[1]["r"]

    # if no stimulus could be extracted
    if not all_stims and not stim_info_flag:
        raise ValueError("No valid stimulus was found in metadata or files. \
            Skipping current file")

    dt = 1. / int(sampling_rate) * 1e3

    # for all segments in file
    for i_seg, seg in enumerate(bl.segments):

        if stim_info_flag:
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
                tamp = [
                    int(stim_info['tamp'][0] / dt),
                    int(stim_info['tamp'][1] / dt)]
            else:
                tamp = [ion, ioff]

            i_unit = stim_info['i_unit']

            if i_unit.lower() == 'a':
                current = current * 1e9  # nA
            elif i_unit.lower() == 'pa':
                current = current * 1e-3  # nA
            elif i_unit.lower() == 'na':
                pass
            else:
                raise Exception(
                    "Unit current not configured!")

            amp = numpy.nanmean(current[tamp[0]:tamp[1]])
            hypamp = numpy.nanmean(current[0:ion])

        else:
            # the following loop is needed because the voltage is not always
            # in the first array of the segment
            voltage = []
            for i_asig, asig in enumerate(seg.analogsignals):
                crr_unit = str(seg.analogsignals[i_asig].units.dimensionality)
                if str(crr_unit.lower()) in common.manageConfig.vu:
                    voltage = numpy.array(asig).astype(numpy.float64).flatten()

            if len(voltage) == 0:
                continue

            t = numpy.arange(len(voltage)) * dt
            ton = all_stims["st"][i_seg]
            toff = all_stims["en"][i_seg]
            ion = int(ton / dt)
            ioff = int(toff / dt)
            amp = numpy.float64(all_stims["crr_val"][i_seg])
            stim_u = all_stims["u"][i_seg]

            # convert stimulus amplitude to nA
            conv_fact = common.manageConfig.conversion_factor('nA', stim_u)

            amp = amp * conv_fact
            if conv_fact != 1:
                stim_u = "nA"

            current = []
            current = numpy.zeros(len(voltage))
            current[ion:ioff] = amp

            # estimate hyperpolarization current
            hypamp = numpy.mean(current[0:ion])

            # 10% distance to measure step current
            iborder = int((ioff - ion) * 0.1)

            # clean voltage from transients
            voltage[ion:ion + int(numpy.ceil(0.4 / dt))] = \
                voltage[ion + int(numpy.ceil(0.4 / dt))]
            voltage[ioff:ioff + int(numpy.ceil(0.4 / dt))] = \
                voltage[ioff + int(numpy.ceil(0.4 / dt))]

        # normalize membrane potential to known value
        # (given in UCL excel sheet)
        if isinstance(v_corr, list):
            if len(v_corr) == 1 and v_corr[0] != 0.0:
                voltage = voltage - numpy.mean(voltage[0:ion]) + v_corr[0]
            elif len(v_corr) - 1 >= idx_file and v_corr[0] != 0.0:
                voltage = voltage - numpy.mean(voltage[0:ion]) \
                    + v_corr[idx_file]
        elif v_corr != 0.0:
            voltage = voltage - numpy.mean(voltage[0:ion]) + v_corr

        voltage = voltage - ljp

        # clip spikes after stimulus so they are not analysed
        voltage[ioff:] = numpy.clip(voltage[ioff:], -300, -40)

        # extract stim traces to be excluded
        [crr_exc, crr_exc_u] = common.manageConfig.get_exclude_values(
            cells[cellname], idx_file)

        if not len(crr_exc) == 0 and any(abs(crr_exc - amp) < 1e-4):
            continue  # llb
        else:
            common.manageDicts.fill_dict_single_trace(
                data=data, voltage=voltage, current=current, dt=dt, t=t,
                ton=ton, toff=toff, amp=amp, hypamp=hypamp,
                filename=filename)
    resp_check = check_validity(data)

    return data


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
                # being here, len(valid_epoch_dicts) == 1, so all the stimulus
                # epochs should be contained in its first and only element
                # (normally zero)

                k = valid_epoch_dicts[0]

                stim_epochs = dictEpochInfoPerDAC[valid_epoch_dicts[0]]

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
                        float(
                            format(
                                stim_epochs[0][
                                    'fEpochLevelInc'], '.3f')) != 0 or
                        (len(stim_ch_info) != 1 or
                            stim_ch_info[0][2] != k)):
                    # return 0 with message
                    return (
                        0, "A stimulus different from the steps has been \
                        detected")
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
                    all_stim_feats = {
                        "ty": [],
                        "st": [],
                        "en": [],
                        "crr_val": [],
                        "u": []
                    }

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

                        # convert unit from bytes to string
                        if sys.version_info[0] >= 3:
                            ustr = str(u, 'utf-8')
                        else:
                            ustr = str(u)

                        all_stim_feats["ty"].append(ty)
                        all_stim_feats["st"].append(st)
                        all_stim_feats["en"].append(en)
                        all_stim_feats["crr_val"].append(crr_val)
                        all_stim_feats["u"].append(ustr)

                    return (1, all_stim_feats)
    else:
        return (0, {})


def sampling_rate_from_header(header):
    """
    Extract sampling rate from the header of the abf.file
    """

    # read version
    version = header['fFileVersionNumber']  # read file version

    if version < 2.:
        # read sampling rate
        nbchannel = header['nADCNumChannels']
        sampling_rate = 1. / (header['fADCSampleInterval'] * nbchannel * 1.e-6)

    elif version >= 2.:
        # read sampling rate
        sampling_rate = 1.e6 / header['protocol']['fADCSequenceInterval']

    return(1, {"r": sampling_rate, "ru": "Hz"})


def get_nbepisod(header):
    # read version
    version = header['fFileVersionNumber']  # read file version

    if version < 2.:
        # read sampling rate
        nbepisod = header['lActualEpisodes']

    elif version >= 2.:
        # read sampling rate
        nbepisod = header['lActualEpisodes']

    return nbepisod


#
def check_validity(data):
    # extract number of traces
    nb_traces = len(data["voltage"])

    # extract number of stimuli
    nb_stims = len(data['current'])

    if nb_traces != nb_stims:
        raise Exception("Number of traces and number of given \
                stimuli are different for file: " + filename)
    else:
        return (1, "")
