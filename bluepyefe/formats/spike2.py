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
import numpy as np
import os
import json
import sys
import quantities
from quantities import ms, s, mV, nA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo import AnalogSignal
from neo.io import Spike2IO
from . import common


def process(config=None,
            filename=None,
            cellname=None,
            expname=None,
            stim_feats=None,
            idx_file=None,
            ljp=0, v_corr=0):

    path = config['path']
    cells = config['cells']

    # initialize data dictionary
    data = common.manageDicts.initialize_data_dict()

    # read metadata file if present
    meta_dict = {}

    f_meta = os.path.join(path, cellname, filename + '_metadata.json')

    #
    if os.path.exists(f_meta):
        with open(f_meta) as json_metadata:
            meta_dict = json.load(json_metadata)
        json_metadata.close()
    else:
        logger.info(
            "No metadata file found for file %s. Skipping file", filename)
        return data

    # read stimulus features if present
    stim_feats = []

    logger.info(" Adding spike2 file %s", filename)

    fullfilename = filename + '.smr'

    logger.info("Reading file :" + fullfilename)

    f = os.path.join(path, cellname, fullfilename)

    # read .smr data
    signals = read_data(f)

    # extract parameters from metadata
    gain, voltage_unit, vm_channel, stimulus_start, stimulus_end, \
        stimulus_time_unit, stim_channel, holding_voltage, stimulus_unit, \
        stimulus_threshold = extract_metadata(meta_dict)

    # check parameters
    if None in (gain, voltage_unit, vm_channel, stimulus_start, stimulus_end,
                stimulus_time_unit, stim_channel, stimulus_unit,
                stimulus_threshold):
        logger.info(
            "Parameters are not valid or not present. Skipping file: " +
            fullfilename
        )
        return data

    # read vm signal
    vm = signals[vm_channel[0]]
    conv_factor = common.manageConfig.conversion_factor("mV", voltage_unit)
    vm = set_units(vm, gain * conv_factor, mV)  # add dimension to signal
    vm = vm.flatten()[::vm_channel[1] + 1]

    # read stim signal
    stim = signals[stim_channel[0]]
    conv_factor = common.manageConfig.conversion_factor("nA", stimulus_unit)
    holding_current = common.manageMetadata.get_holding_current(
        meta_dict, "nA")
    stim = set_units(stim, conv_factor, "nA")  # add dimension to signal

    stim = stim - holding_current["holdcurr"]["value"] * stim.units

    stim = stim.flatten()[::stim_channel[1] + 1]
    stimulus_threshold = stimulus_threshold * conv_factor

    # find edges of the signal where different stimulus steps are located
    step_edges = []
    for idx_sts in range(len(stimulus_start)):
        sts = stimulus_start[idx_sts]
        ste = stimulus_end[idx_sts]
        se = find_stimulus_steps(
            stim.time_slice(sts, ste),
            stimulus_threshold
        )
        if idx_sts == 0:
            step_edges = se
        else:
            se_mag = se.magnitude
            se_unit = se.units
            step_edges_mag = step_edges.magnitude
            mag = np.append(step_edges_mag, se_mag)
            step_edges = mag * se_unit

    if "remove_transient_size" in cells[cellname] and \
            cells[cellname]['remove_transient_size']:
        # remove transient
        vm = remove_transients(
            vm, step_edges, transient_length=cells[cellname]
            ['remove_transient_size'][0][0] * ms
        )

    # extract onsets and offsets from edges
    step_onsets = step_edges[::2]
    step_offsets = step_edges[1::2]

    all_vm = vm.flatten()
    all_stim = stim

    # adding margins to recording chunks
    margins = 50 * ms

    # extract values for individual stimuli
    for onset, offset in zip(step_onsets, step_offsets):

        # slicing current sweep voltage signal
        voltage = all_vm.time_slice(onset - margins, offset + margins)
        voltage = voltage.flatten()

        # slicing current sweep stimulus signal
        stimulus = all_stim.time_slice(
            onset - margins,
            offset + margins
        )

        # extracting stimulus chunks
        stim_no_borders = all_stim.time_slice(
            onset + 0.0 * ms,
            offset - 0.0 * ms
        )

        # extracting time step
        dt = 1. / int(all_vm.sampling_rate) * 1e3

        # time
        t = (voltage.times - voltage.t_start).rescale("ms")

        # extracting onset and offset index from arrays
        ion = time_index(stimulus, onset)
        ioff = time_index(stimulus, offset)

        # convert indices to timestamps
        ton = ion * dt
        toff = ioff * dt

        # duration in ms of signal slice from which to extract the
        # stimulus amplitude
        cut = 5
        amp = extract_amp_from_sig(stim_no_borders, onset, offset, cut)
        amp = np.float64(amp)

        # convert stimulus to string
        amp = np.float64(str("{0:.2f}".format(amp)))

        # creating current signal
        current = []
        current = np.zeros(len(voltage))
        current[ion:ioff] = amp

        # estimate hyperpolarization current
        hypamp = np.mean(current[0:ion])

        # convert voltage from AnalogSignal to list
        voltage = np.array(voltage.tolist()).astype(np.float64)

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

        # extract stim traces to be excluded
        [crr_exc, crr_exc_u] = common.manageConfig.get_exclude_values(
            cells[cellname], idx_file)

        t = t.magnitude.flatten()

        if not len(crr_exc) == 0 and any(abs(crr_exc - amp) < 1e-4):
            continue  # llb
        else:
            common.manageDicts.fill_dict_single_trace(
                data=data, voltage=voltage, current=current, dt=dt, t=t,
                ton=ton, toff=toff, amp=amp, hypamp=hypamp,
                filename=filename)
    return data


def extract_metadata(metadata):
    """
    Extract metadata needed to read .smr traces
    """

    try:
        gain = float(metadata["gain"])
    except Exception as ex:
        logger.error(
            "Error reading 'gain' in metadata or key not present. " +
            "Storing value: None"
        )
        logger.error(ex)
        gain = None

    try:
        stimulus_threshold = float(metadata["stimulus_threshold"][0])
    except Exception as ex:
        logger.error(
            "Error reading 'stimulus_threshold' in metadata or key not " +
            "present. Storing value: None"
        )
        logger.error(ex)
        stimulus_threshold = None

    try:
        holding_voltage = float(metadata["holding_voltage"][0])
    except Exception as ex:
        logger.error(
            "Error reading 'holding_voltage' in metadata or key not " +
            "present. Storing value: '0.0'"
        )
        logger.error(ex)
        holding_voltage = 0.0

    try:
        voltage_unit = metadata["voltage_unit"]
    except Exception as ex:
        logger.error(
            "Error reading 'voltage_unit' in metadata or key not present. " +
            "Storing value: 'mV'"
        )
        logger.error(ex)
        voltage_unit = "mV"

    try:
        stimulus_time_unit = getattr(
            quantities, metadata["stimulus_time_unit"])
    except Exception as ex:
        logger.error(
            "Error reading 'stimulus_time_unit' in metadata or key not " +
            "present. Storing value: 'None'"
        )
        logger.error(ex)
        stimulus_time_unit = None

    try:
        vm_channel = metadata.get("vm_channel", None)
    except Exception as ex:
        logger.error(
            "Error reading 'vm_channel' in metadata or key not present. " +
            "Storing value: 'None'"
        )
        logger.error(ex)
        vm_channel = None

    try:
        if "stimulus_start" not in metadata.keys():
            raise Exception
        else:
            stim_start = metadata["stimulus_start"]
            if type(stim_start) in (float, int):
                stimulus_start = float(stim_start)
            elif type(stim_start) == list and len(stim_start) >= 1:
                stimulus_start = []
                for i in stim_start:
                    stimulus_start.append(i * stimulus_time_unit)
            else:
                raise Exception
    except Exception as ex:
        logger.error(
            "Error reading 'stimulus_start' in metadata, key not present " +
            "or wrong. Storing value: 'None'"
        )
        logger.error(ex)
        stimulus_start = None

    try:
        if "stimulus_end" not in metadata.keys():
            raise Exception
        else:
            stim_end = metadata["stimulus_end"]
            if type(stim_end) in (float, int):
                stimulus_end = float(stim_end)
            elif type(stim_end) == list and len(stim_end) >= 1:
                stimulus_end = []
                for i in stim_end:
                    stimulus_end.append(i * stimulus_time_unit)
            else:
                raise Exception
    except Exception as ex:
        logger.error(
            "Error reading 'stimulus_end' in metadata, key not present or " +
            "wrong. Storing value: 'None'"
        )
        logger.error(ex)
        stimulus_end = None

    try:
        stim_channel = metadata["stim_channel"]
    except Exception as ex:
        logger.error(
            "Error reading 'stim_channel' in metadata or key not present. " +
            "Storing value: None"
        )
        logger.error(ex)
        stim_channel = None

    try:
        stimulus_unit = metadata["stimulus_unit"]
    except Exception as ex:
        logger.error(
            "Error reading 'stimulus_unit' in metadata or key not present." +
            " Storing value: 'nA'"
        )
        logger.error(ex)
        stimulus_unit = None

    return \
        gain, voltage_unit, vm_channel, stimulus_start, stimulus_end, \
        stimulus_time_unit, stim_channel, holding_voltage, stimulus_unit, \
        stimulus_threshold


def set_units(signal, gain, unit):
    """
    Author: Andrew Davison
    """
    # cannot change the units of a signal once created, so we create a new
    # signal from the original data (original signal is dimensionless)
    orig = signal
    sig = AnalogSignal(
        orig.magnitude * gain,
        units=unit,
        sampling_rate=orig.sampling_rate
    )
    sig._copy_data_complement(orig)

    return sig


def find_stimulus_steps(stimulus, stimulus_threshold):
    """
    Authors: Andrew Davison, Luca L. Bologna
    """
    # cut signals into segments so as to overlay responses to different
    # stimulus steps;
    x = np.abs(stimulus.magnitude.flatten()) > stimulus_threshold
    step_edge_indices = np.nonzero(x[1:] ^ x[:-1])[0]

    # does not include step of amplitude zero.
    step_edges = stimulus.times[step_edge_indices]

    return step_edges


def remove_transients(
        vm_trace, stimulus_step_edges, transient_length=1.0 * ms,
        window=5.0 * ms):

    """
    Replace the voltage transients at stimulus onset and offset by the mean of
    the previous few milliseconds.

    This is a rather crude method, and is problematic for offsets if there is
    an action potential just before offset.
    A better approach might be to fit an exponential to the transient.
    Authors: Andrew Davison, Luca L. Bologna
    """
    vm = AnalogSignal(
        vm_trace.magnitude,
        sampling_rate=vm_trace.sampling_rate,
        units=vm_trace.units
    )
    for t_step in stimulus_step_edges:
        length = int(transient_length / vm.sampling_period.rescale('ms'))
        level = vm.time_slice(t_step - window, t_step).mean()
        splice = AnalogSignal(level * np.ones((length,)),
                              sampling_rate=vm.sampling_rate,
                              units=vm.units,
                              t_start=t_step)
        vm.splice(splice)
    return vm


def read_data(filename):
    """
    Author: Andrew Davison
    """
    io = Spike2IO(filename)
    data = io.read()

    return data[0].segments[0].analogsignals


def extract_amp_from_sig(signal, onset, offset, cut):

    """
    Extract amplitude value from signal
    """

    signal_cut = signal.time_slice(onset, onset + cut * ms)
    signal_cut = signal_cut.flatten().magnitude.tolist()
    amp = float(max(signal_cut, key=signal_cut.count))

    return amp


# The following function has been taken from the neo-python package
# Copyright (c) 2010-2018, Neo authors and contributors
# All rights reserved.
def time_index(signal, t):
    """Return the array index corresponding to the time `t`"""
    t = t.rescale(signal.sampling_period.units)
    i = (t - signal.t_start) / signal.sampling_period
    i = int(np.rint(i.magnitude))

    return i
