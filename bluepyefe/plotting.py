"""Plotting functions"""

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
import math
import pathlib
from itertools import cycle

import matplotlib.pyplot as plt
import numpy

logger = logging.getLogger(__name__)


def _save_fig(directory, filename):
    dir_path = pathlib.Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    fig_path = dir_path / filename

    plt.savefig(fig_path, dpi=80)
    plt.close("all")
    plt.clf()


def _get_colors_markers_wheels(cells):
    colors = cycle(["C{}".format(i) for i in range(10)])
    markers = cycle(["o", "v", "^", "s", "*", "d", "x"])

    cell_names = [c.name for c in cells]

    colors = {n: next(colors) for n in cell_names}
    markers = {n: next(markers) for n in cell_names}

    return colors, markers


def _plot_legend(colors, markers, output_dir):
    """Draw a separate legend figure"""

    if not len(colors):
        raise Exception("Plot legend needs an non-empty colors dictionary.")
    if not len(markers):
        raise Exception("Plot legend needs an non-empty markers dictionary.")

    ncols = math.ceil(numpy.sqrt(len(colors)))

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

    _save_fig(output_dir, "legend.pdf")

    return fig, axs


def plot_cell_recordings(cell, protocol_name, output_dir):
    """Plot the recordings of a cell for a protocol"""

    recordings = cell.get_recordings_by_protocol_name(protocol_name)

    if not len(recordings):
        return None, None

    recordings_amp = [rec.amp for rec in recordings]
    recordings = [recordings[k] for k in numpy.argsort(recordings_amp)]

    nbr_x = 10.0
    nbr_y = 2 * math.ceil(len(recordings) / nbr_x)
    figsize = [3.0 + 1.7 * int(nbr_x), 2.2 * nbr_y]
    fig, axs = plt.subplots(
        int(nbr_y), int(nbr_x), figsize=figsize, squeeze=False
    )

    for i, rec in enumerate(recordings):

        xpos = 2 * int(i / nbr_x)
        ypos = i % int(nbr_x)

        title = "Amp = {:.03f}".format(rec.amp)

        if rec.amp_rel is not None:
            title += " ({:.01f}%)".format(rec.amp_rel)

        if rec.id is not None:
            title += "\nid: {}".format(rec.id)

        if rec.repetition is not None:
            title += "\nRepetition: {}".format(rec.repetition)

        axs[xpos][ypos].set_title(title, size="x-small")

        axs[xpos][ypos].plot(rec.t, rec.current, c="C0")

        gen_t, gen_i = rec.generate()
        axs[xpos][ypos].plot(gen_t, gen_i, c="C1", ls="--")

        axs[xpos + 1][ypos].plot(rec.t, rec.voltage, c="C0")

        if ypos == 0:
            axs[xpos][ypos].set_ylabel("Current (nA)")
        if ypos == 0:
            axs[xpos + 1][ypos].set_ylabel("Voltage (mV)")

        axs[xpos][ypos].tick_params(axis="both", which="major", labelsize=8)
        axs[xpos][ypos].tick_params(axis="both", which="minor", labelsize=6)
        axs[xpos + 1][ypos].tick_params(
            axis="both", which="major", labelsize=8
        )
        axs[xpos + 1][ypos].tick_params(
            axis="both", which="minor", labelsize=6
        )

    fig.text(0.5, 0.04, "Time (ms)", ha="center")

    fig.suptitle("Cell: {}, Experiment: {}".format(cell.name, protocol_name))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = "{}_{}_recordings.pdf".format(cell.name, protocol_name)
    dirname = pathlib.Path(output_dir) / cell.name
    _save_fig(dirname, filename)

    return fig, axs


def plot_all_recordings(cells, output_dir):
    """Plot recordings for all cells and all protocols"""

    for cell in cells:
        for protocol_name in cell.get_protocol_names():
            plot_cell_recordings(cell, protocol_name, output_dir)


def plot_efeature(
        cells,
        efeature,
        protocol_name,
        output_dir,
        protocols=[],
        key_amp="amp",
        colors=None,
        markers=None,
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

                ax.errorbar(
                    x_key,
                    target.mean,
                    yerr=target.std,
                    marker="o",
                    elinewidth=0.7,
                    markersize=3.0,
                    c="gray",
                    zorder=100,
                    alpha=0.6,
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
    output_dir,
    protocols=[],
    key_amp="amp",
    colors=None,
    markers=None
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
            protocols=[],
            key_amp=key_amp,
            colors=colors,
            markers=markers,
        )


def plot_individual_efeatures(
    cells, protocols, output_dir, colors=None, markers=None, key_amp="amp"
):

    if not colors or not markers:
        colors, markers = _get_colors_markers_wheels(cells)

    for cell in cells:

        for protocol_name in cell.get_protocol_names():

            _ = plot_efeatures(
                cells=[cell],
                protocol_name=protocol_name,
                output_dir=output_dir,
                protocols=protocols,
                key_amp=key_amp,
                colors=colors,
                markers=markers,
            )


def plot_grouped_efeatures(
    cells, protocols, output_dir, colors=None, markers=None, key_amp="amp"
):

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
        )

    _ = _plot_legend(colors, markers, output_dir)


def plot_all_recordings_efeatures(cells, protocols, output_dir):
    colors, markers = _get_colors_markers_wheels(cells)

    plot_all_recordings(cells, output_dir)

    for key_amp in ["amp", "amp_rel"]:
        plot_individual_efeatures(
            cells,
            protocols,
            output_dir,
            colors=colors,
            markers=markers,
            key_amp=key_amp,
        )
        plot_grouped_efeatures(
            cells,
            protocols,
            output_dir,
            colors=colors,
            markers=markers,
            key_amp=key_amp,
        )
