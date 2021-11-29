"""Functions related to the computation of the rheobase"""

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
import numpy

logger = logging.getLogger(__name__)


def compute_rheobase_absolute(cell, protocols_rheobase, spike_threshold=1):
    """ Compute the rheobase by finding the smallest current amplitude
    triggering at least one spike.

    Args:
        cell (Cell): cell for which to compute the rheobase
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        spike_threshold (int): number of spikes above which a recording
            is considered to compute the rheobase.
    """

    amps = []

    for i, rec in enumerate(cell.recordings):
        if rec.protocol_name in protocols_rheobase:
            if rec.spikecount is not None and \
                    rec.spikecount >= spike_threshold:

                if rec.amp < 0.01:
                    logger.warning(
                        f"A recording of cell {cell.name} protocol "
                        f"{rec.protocol_name} shows spikes at a "
                        "suspiciously low current in a trace from file"
                        f" {rec.files}. Check that the ton and toff are"
                        "correct or for the presence of unwanted spikes."
                    )

                amps.append(rec.amp)

    if len(amps):
        cell.rheobase = numpy.min(amps)


def compute_rheobase_majority_bin(cell, protocols_rheobase, min_step=0.01, majority=0.5):
    """ Compute the rheobase by finding the smallest current amplitude
    triggering at least 1 spikes in the majority (default 50%) of the recordings.

    Args:
        cell (Cell): cell for which to compute the rheobase
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        min_step (float): minimum step above which amplitudes can be
            considered as separate steps
        majority (float): the proportion of sweeps with spike_threshold
            spikes to consider the target amplitude as rheobase
    """

    amps = []
    spike_counts = []

    for i, rec in enumerate(cell.recordings):
        if rec.protocol_name in protocols_rheobase:
            if rec.spikecount is not None:

                amps.append(rec.amp)
                spike_counts.append(rec.spikecount)

                if rec.amp < 0.01 and rec.spikecount >= 1:
                    logger.warning(
                        f"A recording of cell {cell.name} protocol "
                        f"{rec.protocol_name} shows spikes at a "
                        "suspiciously low current in a trace from file"
                        f" {rec.files}. Check that the ton and toff are"
                        "correct or for the presence of unwanted spikes."
                    )

    bins = numpy.arange(min(amps), max(amps), min_step)
    bins_of_amps = numpy.digitize(amps, bins, right=False)

    for i, bin in enumerate(bins):

        spikes = [spike_counts[j] for j, idx in enumerate(bins_of_amps) if idx == i]
        perc_spiking = numpy.mean([bool(s) for s in spikes])

        if perc_spiking >= majority:
            cell.rheobase = bin + (min_step / 2.)
            return
