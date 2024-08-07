"""Plotting functions"""

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
import pathlib
from itertools import cycle
from functools import partial

import matplotlib.pyplot as plt
import numpy

logger = logging.getLogger(__name__)


def _save_fig(directory, filename):
    """Save a matplotlib figure.

    Args:
        directory (str): path of the directory in which to save the figure.
        filename (str): name of the file in which to save the figure.
    """

    dir_path = pathlib.Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    fig_path = dir_path / filename

    plt.savefig(fig_path, dpi=80, pad_inches=0)
    plt.close("all")
    plt.clf()


def _get_colors_markers_wheels(cells):
    """Generate a color and a marker dictionary unique to each cell.

    Args:
        cells (list of Cells): list of the cells
    """

    colors = cycle(["C{}".format(i) for i in range(10)])
    markers = cycle(["o", "v", "^", "s", "*", "d", "x"])

    cell_names = [c.name for c in cells]

    colors = {n: next(colors) for n in cell_names}
    markers = {n: next(markers) for n in cell_names}

    return colors, markers


def _plot_legend(colors, markers, output_dir, show=False):
    """Draw a separate legend figure"""

    if not len(colors):
        raise Exception("Plot legend needs an non-empty colors dictionary.")
    if not len(markers):
        raise Exception("Plot legend needs an non-empty markers dictionary.")

    ncols = numpy.ceil(numpy.sqrt(len(colors)))

    fig, axs = plt.subplots(1, figsize=(6, 6), squeeze=False)

    for i, (cellname, c, m) in enumerate(
        zip(colors.keys(), colors.values(), markers.values())
    ):
        axs[0, 0].scatter(
            x=2.5 * int(i / ncols), y=0.5 * (i % ncols), c=c, marker=m, s=8.0
        )

        axs[0, 0].text(
            x=2.5 * int(i / ncols) + 0.1,
            y=0.5 * (i % ncols) - 0.08,
            s=cellname,
            fontsize=5,
        )

    axs[0, 0].set_xlim(-1.0, 2.0 * ncols + 1)
    axs[0, 0].set_ylim(-1.0, ncols + 1)

    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 0].set_frame_on(False)

    if show:
        fig.show()

    if output_dir:
        _save_fig(output_dir, "legend.pdf")

    return fig, axs


def _plot(cell, output_dir, show=False):
    cell.plot_all_recordings(output_dir, show=show)


def plot_all_recordings(cells, output_dir, show=False, mapper=map):
    """Plot recordings for all cells and all protocols"""
    if mapper == map:
        # For the built-in map(), ensure immediate evaluation as it returns a lazy iterator
        # which won't execute the function until iterated over. Converting to a list forces this iteration.
        list(mapper(partial(_plot, output_dir=output_dir, show=show), cells))
    else:
        mapper(partial(_plot, output_dir=output_dir, show=show), cells)


def plot_efeature(
    cells,
    efeature,
    protocol_name,
    output_dir,
    protocols=[],
    key_amp="amp",
    colors=None,
    markers=None,
    show=False,
    show_targets=True,
):
    """Plot one efeature for a protocol"""

    if not cells:
        logger.warning("In plot_efeature, no cells object to plot.")
        return None

    if not colors or not markers:
        colors, markers = _get_colors_markers_wheels(cells)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    has_data = False

    for cell in cells:

        x, y = [], []
        for rec in cell.get_recordings_by_protocol_name(protocol_name):
            if hasattr(rec, key_amp) and getattr(rec, key_amp):
                if efeature in rec.efeatures:
                    x.append(getattr(rec, key_amp))
                    y.append(rec.efeatures[efeature])

        if y:
            has_data = True

        ax.scatter(
            x, y, c=colors[cell.name], marker=markers[cell.name], s=5.0
        )

        for protocol in protocols:
            if protocol.name == protocol_name:

                if key_amp == "amp_rel":
                    x_key = protocol.amplitude
                elif key_amp == "amp":
                    x_key = numpy.mean([t.amp for t in protocol.recordings])

                target = next(
                    (t for t in protocol.feature_targets if
                     t.efel_feature_name == efeature),
                    None
                )
                if target is not None and show_targets:
                    ax.errorbar(
                        x_key,
                        target.mean,
                        yerr=target.std,
                        marker="o",
                        elinewidth=0.8,
                        markersize=3.0,
                        c="black",
                        zorder=100,
                        alpha=0.8,
                        capsize=2
                    )

    if not has_data:
        return

    if key_amp == "amp_rel":
        ax.set_xlabel(
            "Relative step amplitude" " ($I/I_{thresh}$)", size="x-large"
        )
    elif key_amp == "amp":
        ax.set_xlabel("Step amplitude (nA)", size="x-large")
    ax.set_ylabel(efeature, size="x-large")
    ax.set_title(
        f"Protocol: {protocol_name}, EFeature: {efeature}", size="xx-large"
    )

    if show:
        fig.show()

    if output_dir:
        if len(cells) == 1:
            filename = "{}_{}_{}_{}.pdf".format(
                cells[0].name, protocol_name, efeature, key_amp
            )
            dirname = pathlib.Path(output_dir) / cells[0].name
        else:
            filename = "{}_{}_{}.pdf".format(protocol_name, efeature, key_amp)
            dirname = pathlib.Path(output_dir)

    _save_fig(dirname, filename)


def plot_efeatures(
    cells,
    protocol_name,
    output_dir=None,
    protocols=[],
    key_amp="amp",
    colors=None,
    markers=None,
    show=False,
    show_targets=True,
):
    """
    Plot the efeatures of a cell or a group of cells versus current amplitude
    or relative current amplitude.
    """

    if not cells:
        logger.warning("In plot_efeatures, no cells object to plot.")
        return None, None

    efeatures = set()
    for p in protocols:
        if p.name == protocol_name:
            efeatures.update([t.efel_feature_name for t in p.feature_targets])

    for fi, efeature in enumerate(efeatures):
        plot_efeature(
            cells,
            efeature,
            protocol_name,
            output_dir,
            protocols=protocols,
            key_amp=key_amp,
            colors=colors,
            markers=markers,
            show=show,
            show_targets=show_targets,
        )


def _plot_ind(cell, output_dir, protocols, key_amp, colors, markers, show):
    for protocol_name in cell.get_protocol_names():
        _ = plot_efeatures(
            cells=[cell],
            protocol_name=protocol_name,
            output_dir=output_dir,
            protocols=protocols,
            key_amp=key_amp,
            colors=colors,
            markers=markers,
            show=show,
            show_targets=False
        )


def plot_individual_efeatures(
    cells,
    protocols,
    output_dir=None,
    colors=None,
    markers=None,
    key_amp="amp",
    show=False,
    mapper=map,
):
    """Generate efeatures plots for all each cell individually"""

    if not colors or not markers:
        colors, markers = _get_colors_markers_wheels(cells)

    mapper(
        partial(
            _plot_ind,
            output_dir=output_dir,
            protocols=protocols,
            key_amp=key_amp,
            colors=colors,
            markers=markers,
            show=show,
        ),
        cells,
    )


def plot_grouped_efeatures(
    cells,
    protocols,
    output_dir=None,
    colors=None,
    markers=None,
    key_amp="amp",
    show=False
):
    """Generate plots for each efeature across all cells."""

    if not cells:
        logger.warning("In plot_grouped_efeatures, no cells object to plot.")
        return

    if not colors or not markers:
        colors, markers = _get_colors_markers_wheels(cells)

    protocol_names = set([p.name for p in protocols])

    for protocol_name in protocol_names:

        _ = plot_efeatures(
            cells=cells,
            protocol_name=protocol_name,
            output_dir=output_dir,
            protocols=protocols,
            key_amp=key_amp,
            colors=colors,
            markers=markers,
            show=show
        )

    _ = _plot_legend(colors, markers, output_dir, show=show)


def plot_impedance(cell, output_dir, efel_settings):
    """Plots the impedance."""
    from scipy.ndimage.filters import gaussian_filter1d

    dt = 0.1
    Z_max_freq = 50.0
    if efel_settings is not None:
        dt = efel_settings.get("interp_step", dt)
        Z_max_freq = efel_settings.get("impedance_max_freq", Z_max_freq)

    for protocol_name in cell.get_protocol_names():
        if "sinespec" in protocol_name.lower():
            recordings = cell.get_recordings_by_protocol_name(protocol_name)
            for i, rec in enumerate(recordings):
                voltage = rec.voltage
                current = rec.current

                efel_vals = rec.call_efel(
                    [
                        "voltage_base",
                        "steady_state_voltage_stimend",
                        "current_base",
                        "steady_state_current_stimend",
                    ],
                    efel_settings
                )
                if efel_vals[0]["voltage_base"] is not None:
                    holding_voltage = efel_vals[0]["voltage_base"][0]
                else:
                    holding_voltage = efel_vals[0]["steady_state_voltage_stimend"][0]
                if efel_vals[0]["current_base"] is not None:
                    holding_current = efel_vals[0]["current_base"][0]
                else:
                    holding_current = efel_vals[0]["steady_state_current_stimend"][0]

                normalized_voltage = voltage - holding_voltage
                normalized_current = current - holding_current

                fft_volt = numpy.fft.fft(normalized_voltage)
                fft_cur = numpy.fft.fft(normalized_current)
                if any(fft_cur) == 0:
                    continue
                # convert dt from ms to s to have freq in Hz
                freq = numpy.fft.fftfreq(len(normalized_voltage), d=dt / 1000.)
                Z = fft_volt / fft_cur
                norm_Z = abs(Z) / max(abs(Z))
                select_idxs = numpy.swapaxes(
                    numpy.argwhere((freq > 0) & (freq <= Z_max_freq)), 0, 1
                )[0]
                smooth_Z = gaussian_filter1d(norm_Z[select_idxs], 10)

                filename = "{}_{}_{}_{}_impedance.pdf".format(
                    cell.name, protocol_name, i, rec.name
                )
                dirname = pathlib.Path(output_dir) / cell.name / "impedance"

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(freq[select_idxs], smooth_Z)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("normalized Z")
                fig .suptitle(f"Impedance for {rec.name}\nfor cell {cell.name}")
                _save_fig(dirname, filename)


def plot_all_impedances(cells, output_dir, mapper=map, efel_settings=None):
    """Plot recordings for all cells and all protocols"""
    if mapper == map:
        # For the built-in map(), ensure immediate evaluation as it returns a lazy iterator
        # which won't execute the function until iterated over. Converting to a list forces this iteration.
        list(mapper(partial(plot_impedance, output_dir=output_dir, efel_settings=efel_settings), cells))
    else:
        mapper(partial(plot_impedance, output_dir=output_dir, efel_settings=efel_settings), cells)


def plot_all_recordings_efeatures(
    cells, protocols, output_dir=None, show=False, mapper=map, efel_settings=None
):
    """Generate plots for all recordings and efeatures both for individual
    cells and across all cells."""

    colors, markers = _get_colors_markers_wheels(cells)

    plot_all_recordings(cells, output_dir, mapper=mapper)
    plot_all_impedances(cells, output_dir, mapper=map, efel_settings=efel_settings)

    for key_amp in ["amp", "amp_rel"]:
        plot_individual_efeatures(
            cells,
            protocols,
            output_dir,
            colors=colors,
            markers=markers,
            key_amp=key_amp,
            show=show,
            mapper=mapper,
        )
        plot_grouped_efeatures(
            cells,
            protocols,
            output_dir,
            colors=colors,
            markers=markers,
            key_amp=key_amp,
            show=show
        )
