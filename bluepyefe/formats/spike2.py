from neo import io
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
import numpy as np
import os
import json
import pprint
import sys
import quantities
from quantities import ms
from quantities import s
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo import AnalogSignal
from neo.io import Spike2IO


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common

def process(config=None,
            filename=None,
            cellname=None,
            expname=None,
            stim_feats=None,
            idx_file=None,
            ljp=0, v_corr=0):

    try:
        common.manageConfig.check_config(config, "spike2")
    except ValueError as error:
        logging.info(error)

    path = config['path']
    cells = config['cells']
    features = config['features']
    options = config['options']

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
        logger.debug("No metadata file found for file %s. \
                Skipping file", filename)
        print("No metadata found. Skipping file " + filename)
        print("Returning for error")
        return data

    # read stimulus features if present
    stim_feats = []

    logger.debug(" Adding spike2 file %s", filename)

    fullfilename = filename + '.smr'
    
    print("Reading file :" + fullfilename)

    f = os.path.join(path, cellname, fullfilename)
    
    # read .smr data
    signals = read_data(f)

    # extract parameters from metadata
    gain, voltage_unit, vm_channel, stimulus_start, stimulus_end, \
        stimulus_time_unit, stim_channel, holding_voltage, stim_unit = \
            extract_metadata(meta_dict)

    # check parameters
    if None in (gain, vm_channel, stimulus_start, stimulus_end, stimulus_time_unit):
        print("Parameters are not valid or not present. Skipping file: " + \
                fullfilename)
        return data

    # add dimension to signal
    signals[vm_channel] = set_voltage_units(signals[vm_channel], gain, \
            voltage_unit)
    
    # find edges of the entire signal where different stimulus steps are located
    step_edges = find_stimulus_steps(signals[stim_channel].time_slice(stimulus_start, stimulus_end))


    # fill gaps in step_edges looking for 0 amplitude stimulus
    step_edges = fill_zero_stim_edges(step_edges)

    if "remove_transient_size" in cells[cellname] and \
            cells[cellname]['remove_transient_size']:
        # remove transient
        signals[vm_channel] = remove_transients(signals[vm_channel], \
                step_edges, transient_length=cells[cellname]['remove_transient_size'][0][0] * ms)

    # extract onsets and offsets from edges
    step_onsets = step_edges[::2]
    step_offsets = step_edges[1::2]

    all_vm = signals[vm_channel]
    all_stim = signals[stim_channel]

    # adding margins to recording chunks
    margins = 50 * ms

    # extract values for individual stimuli
    for onset, offset in zip(step_onsets, step_offsets):

        # slicing current sweep voltage signal
        voltage = all_vm.time_slice(onset - margins, offset + margins) 

        # slicing current sweep stimulus signal
        stim = signals[stim_channel].time_slice(onset - margins, offset + margins)

        # extracting stimulus chunks
        stim_no_borders = signals[stim_channel].time_slice(onset + 0.0 * ms, \
                offset - 0.0 * ms)

        # extracting time step
        dt = 1./int(all_vm.sampling_rate) * 1e3

        # time
        t = (voltage.times-voltage.t_start).rescale("ms")

        # extracting onset and offset index from arrays
        ion = time_index(stim, onset)
        ioff = time_index(stim, offset)
        
        # convert indices to timestamps
        ton = ion * dt
        toff = ioff * dt

        cut = 5 # duration in ms of signal slice from which to extract the stimulus amplitude 
        amp = extract_amp_from_sig(stim_no_borders, onset, offset, cut)
        amp = np.float64(amp)

        # convert stimulus to nA
        conv_fact = common.manageConfig.conversion_factor('nA', stim_unit)
        amp = amp * conv_fact 
        amp = np.float64(str("{0:.2f}".format(amp)))
        if conv_fact != 1:
            stim_u = "nA"

        # creating current signal
        current = []
        current = np.zeros(len(voltage))
        current[ion:ioff] = amp

        # estimate hyperpolarization current
        hypamp = np.mean( current[0:ion] )

        # convert voltage from AnalogSignal to list
        voltage = np.array(voltage.tolist()).astype(np.float64)

        # clean voltage from transients

        # normalize membrane potential to known value (given in UCL excel sheet)
        if v_corr:
            if len(v_corr) == 1 and v_corr[0] != 0.0:
                voltage = voltage - np.mean(voltage[0:ion]) + v_corr[0]
            elif len(v_corr) - 1 >= idx_file and v_corr[idx_file] != 0.0:
                voltage = voltage - np.mean(voltage[0:ion]) \
                        + v_corr[idx_file]

        voltage = voltage - ljp

        # clip spikes after stimulus so they are not analysed
        voltage[ioff:] = np.clip(voltage[ioff:], -300, -40)

        # extract stim traces to be excluded
        [crr_exc, crr_exc_u] = common.manageConfig.get_exclude_values( \
                cells[cellname],idx_file)
        
        if not len(crr_exc) == 0 and any(abs(crr_exc - amp) < 1e-4):
            continue # llb
        else:
            common.manageDicts.fill_dict_single_trace( \
                    data=data, voltage=voltage, current=current, dt=dt, t=t, \
                    ton=ton, toff=toff, amp=amp, hypamp=hypamp, \
                    filename=filename)

    return data



def extract_metadata(metadata):
    """
    Extract metadata needed to read .smr traces
    """

    try:
        gain = float(metadata["gain"])
    except:
        print("Error reading 'gain' in metadata or key not present. " + \
                "Storing value: 'None'")
        gain = None
    
    try:
        holding_voltage = float(metadata["holding_voltage"][0])
    except:
        print("Error reading 'holding_voltage' in metadata or key not present. " + \
                "Storing value: '0.0'")
        holding_voltage = 0.0

    try:
        voltage_unit = metadata["holding_voltage_unit"]
    except:
        print("Error reading 'voltage_unit' in metadata or key not present. " + \
                "Storing value: 'mV'")
        voltage_unit = "mV"

    try:
        stimulus_time_unit = getattr(quantities, metadata["stimulus_time_unit"])
    except:
        print("Error reading 'stimulus_time_unit' in metadata or key not present. " + \
                "Storing value: 'None'")
        stimulus_time_unit = None

    try:
        vm_channel = metadata.get("vm_channel", None)   # this entry added manually to metadata files
    except:
        print("Error reading 'vm_channel' in metadata or key not present. " + \
                "Storing value: 'None'")
        vm_channel = None

    try:
        stimulus_start = metadata["stimulus_start"][0] * stimulus_time_unit   # these entries modified manually
    except:
        print("Error reading 'stimulus_start' in metadata or key not present. " + \
                "Storing value: 'None'")
        stimulus_start = None

    try:
        stimulus_end = metadata["stimulus_end"][0] * stimulus_time_unit       # in metadata files
    except:
        print("Error reading 'stimulus_end' in metadata or key not present. " + \
                "Storing value: 'None'")
        stimulus_end = None

    try:
        stim_channel = metadata["stim_channel"]   # this entry added manually to metadata files
    except:
        print("Error reading 'stim_channel' in metadata or key not present. " + \
                "Storing value: 3")
        stim_channel = 3

    try:
        stim_unit = metadata["stimulus_unit"]   # this entry added manually to metadata files
    except:
        print("Error reading 'stimulus_unit' in metadata or key not present. " + \
                "Storing value: 'nA'")
        stim_unit = 'nA'


    return gain, voltage_unit, vm_channel, stimulus_start, stimulus_end, \
            stimulus_time_unit, stim_channel, holding_voltage, stim_unit


def set_voltage_units(signal, gain, voltage_unit):
    """
    Author: Andrew Davison
    """
    # cannot change the units of a signal once created, so we create a new signal
    # from the original data (original signal is dimensionless)
    orig = signal
    vm = AnalogSignal(orig.magnitude * gain,
                      units=voltage_unit,
                      sampling_rate=orig.sampling_rate)
    vm._copy_data_complement(orig)

    return vm


def find_stimulus_steps(stimulus):
    """
    Author: Andrew Davison
    """
    # cut signals into segments so as to overlay responses to different stimulus steps
    x = np.abs(stimulus.magnitude.flatten()) > 0.05  # arbitrary threshold based on visual inspection
    step_edge_indices = np.nonzero(x[1:] ^ x[:-1])[0]
    step_edges = stimulus.times[step_edge_indices]  # does not include step of amplitude zero.
    
    return step_edges


def remove_transients(vm_trace, stimulus_step_edges, transient_length=1.0 * ms, window=5.0 * ms):
    """
    Replace the voltage transients at stimulus onset and offset by the mean of the previous few milliseconds.

    This is a rather crude method, and is problematic for offsets if there is an action potential just before offset.
    A better approach might be to fit an exponential to the transient.
    Author: Andrew Davison
    """
    for t_step in stimulus_step_edges:
        length = int(transient_length / vm_trace.sampling_period.rescale('ms'))
        level = vm_trace.time_slice(t_step - window, t_step).mean()
        splice = AnalogSignal(level * np.ones((length,)),
                              sampling_rate=vm_trace.sampling_rate,
                              units=vm_trace.units,
                              t_start=t_step)
        vm_trace.splice(splice)
    return vm_trace


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
    amp = float(max(signal_cut.tolist(), key=signal_cut.tolist().count)[0])
    
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



def fill_zero_stim_edges(step_edges):
    """
    Check time differences between consecutive stimuli onset. 
    If only one stimulus onset is twice delayed with respect to all the remaining
    fill the stimulus onsets/offsets list with the appropriate values, assuming
    a zero stimuli was applied in that period
    """

    x = step_edges
    xdiff = [float(x[n])-float(x[n-2]) for n in range(2,len(x))[::2]]

    # extract the most common value in the array
    stim_time_diff = float(max(xdiff, key=xdiff.count))

    # create dictionary with all time differences
    diff_dict = dict((x,xdiff.count(x)) for x in set(xdiff))
    
    # if more than two values are found skip this step
    if len(diff_dict) > 2:
        print("Stimulus intervals are not regular. Skipping eventual zero amplitude stimulus reconstruction")
        return step_edges
    else:
        if not diff_dict or len(diff_dict) == 1:
            print("No jump in stimulus onset values. Skipping eventual zero amplitude stimulus reconstruction")
            return step_edges
        for k in diff_dict:
            if k is not stim_time_diff:
                if diff_dict[k] != 1:
                    print("More than one big 'hole' in stimulus amplitudes. Skipping eventual zero amplitude stimulus reconstruction")
                    return step_edges
                else:
                    if abs(k - 2 * stim_time_diff) < stim_time_diff/10.0:
                        time_jump = k
                        break
                    else:
                        print("Irregular jump into stimulus timing. Skipping eventual zero amplitude stimulus reconstruction")
                        return step_edges

    idx_array = range(len(step_edges))[2::2]
    for idx in idx_array:
        edge = step_edges[idx]
        if edge - step_edges[idx-2] == time_jump * edge.units:
            first_chunk = np.array(step_edges[:idx])
            zero_edge_start = np.float64(step_edges[idx-2] + stim_time_diff * step_edges.units)
            zero_edge_end = zero_edge_start + np.float64(step_edges[idx-1] - step_edges[idx-2])
            zero_chunk = [zero_edge_start, zero_edge_end]
            last_chunk = np.array(step_edges[idx:])

            step_edges = np.append(first_chunk, np.append(zero_chunk, last_chunk))
            step_edges = step_edges * edge.units
            break

    return step_edges
