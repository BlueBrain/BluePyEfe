"""Functions related to the computation of the rheobase"""

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
import numpy

logger = logging.getLogger(__name__)


def _get_list_spiking_amplitude(cell, protocols_rheobase):
    """Return the list of sorted list of amplitude that triggered at least
    one spike"""

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

    if amps:
        amps, spike_counts = zip(*sorted(zip(amps, spike_counts)))

    return amps, spike_counts


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

    amps, spike_counts = _get_list_spiking_amplitude(cell, protocols_rheobase)

    for amp, spike_count in zip(amps, spike_counts):
        if spike_count >= spike_threshold:
            cell.rheobase = amp
            break


def compute_rheobase_flush(cell, protocols_rheobase, flush_length=1, upper_bound_spikecount=None):
    """ Compute the rheobase by finding the smallest current amplitude that:
        1. Triggered at least one spike
        2. Is followed by flush_length other traces that also trigger spikes.
    The advantage of this strategy is that it ignores traces showing spurious
    spikes at low amplitudes.
    Args:
        cell (Cell): cell for which to compute the rheobase
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        flush_length (int): number of traces that needs to show spikes for
            the candidate trace to be considered the rheobase.
        upper_bound_spikecount (int): if the spikecount of a recording is higher
            than this number, the recording will not trigger the start of a flush
    """

    amps, spike_counts = _get_list_spiking_amplitude(cell, protocols_rheobase)

    for i, amp in enumerate(amps):

        # We missed the threshold
        if upper_bound_spikecount is not None:
            if spike_counts[i] > upper_bound_spikecount:
                break

        if spike_counts[i]:

            end_flush = min(i + 1 + flush_length, len(amps))

            if (
                    numpy.count_nonzero(spike_counts[i + 1:end_flush]) == flush_length
            ):
                cell.rheobase = amp
                break


def compute_rheobase_majority_bin(
    cell, protocols_rheobase, min_step=0.01, majority=0.5
):
    """ Compute the rheobase by finding the smallest current amplitude
    triggering at least 1 spikes in the majority (default 50%) of the
    recordings.

    Args:
        cell (Cell): cell for which to compute the rheobase
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
        min_step (float): minimum step above which amplitudes can be
            considered as separate steps
        majority (float): the proportion of sweeps with spike_threshold
            spikes to consider the target amplitude as rheobase
    """

    amps, spike_counts = _get_list_spiking_amplitude(cell, protocols_rheobase)

    bins = numpy.arange(min(amps), max(amps), min_step)
    bins_of_amps = numpy.digitize(amps, bins, right=False)

    for i, bin in enumerate(bins):

        spikes = [
            spike_counts[j] for j, idx in enumerate(bins_of_amps) if idx == i
        ]
        perc_spiking = numpy.mean([bool(s) for s in spikes])

        if perc_spiking >= majority:
            cell.rheobase = bin + (min_step / 2.)
            break


def compute_rheobase_interpolation(cell, protocols_rheobase):
    """ Compute the rheobase by fitting the reverse IF curve and finding the
    intersection with the line x = 1.

    Args:
        cell (Cell): cell for which to compute the rheobase
        protocols_rheobase (list): names of the protocols that will be
            used to compute the rheobase of the cells. E.g: ['IDthresh'].
    """

    amps, spike_counts = _get_list_spiking_amplitude(cell, protocols_rheobase)

    # Remove the excess zeros
    idx = next(
        (i for i in range(len(amps) - 1) if not spike_counts[i] and spike_counts[i + 1]),
        None
    )
    if idx is None:
        return
    amps = amps[idx:]
    spike_counts = spike_counts[idx:]

    if amps:

        try:
            fit = numpy.poly1d(numpy.polyfit(spike_counts, amps, deg=1))
        except:
            logger.error(
                f"Rheobase interpolation did not converge on cell {cell.name}"
            )
            return

        cell.rheobase = fit(1)
