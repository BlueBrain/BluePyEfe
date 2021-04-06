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


import math
import pathlib
from itertools import cycle

import matplotlib.pyplot as plt
import numpy

DPI = 100
PLOT_PER_COLUMN = 5


def _save_fig(directory, filename):
    dir_path = pathlib.Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    fig_path = dir_path / filename

    plt.savefig(fig_path, dpi=DPI)
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

    ncols = math.ceil(numpy.sqrt(len(colors)))

    fig, axs = plt.subplots(1, figsize=(6, 6), squeeze=False)

    for i, (cellname, c, m) in enumerate(
        zip(colors.keys(), colors.values(), markers.values())
    ):
        axs[0, 0].scatter(
            x=2.0 * int(i / ncols), y=i % ncols, c=c, marker=m, s=8.0
        )

        axs[0, 0].text(
            x=2.0 * int(i / ncols) + 0.1,
            y=(i % ncols) - 0.08,
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

        axs[xpos][ypos].set_title(title, size="small")

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


def plot_efeatures(
    cells,
    protocol_name,
    efeatures,
    output_dir,
    protocols=[],
    key_amp="amp",
    colors=None,
    markers=None,
):
    """
    Plot the efeatures of a cell or a group of cells versus current amplitude
    or relative current amplitude.
    """

    if not colors or not markers:
        colors, markers = _get_colors_markers_wheels(cells)

    figsize = [
        3.0 + 1.7 * int(len(efeatures) / PLOT_PER_COLUMN),
        2.0 * PLOT_PER_COLUMN,
    ]

    fig, axs = plt.subplots(
        PLOT_PER_COLUMN,
        math.ceil(len(efeatures) / PLOT_PER_COLUMN),
        figsize=figsize,
        squeeze=False,
    )

    xpos = None
    for fi, efeature in enumerate(efeatures):

        xpos = fi % PLOT_PER_COLUMN
        ypos = int(fi / PLOT_PER_COLUMN)

        for cell in cells:

            x, y = [], []

            for rec in cell.get_recordings_by_protocol_name(protocol_name):

                try:
                    x.append(getattr(rec, key_amp))
                except ObjectDoesNotExist:
                    raise Exception(
                        "The key_amp you are trying to use does not exist"
                        " for this protocol."
                    )

                y.append(rec.efeatures[efeature])

            axs[xpos][ypos].scatter(
                x, y, c=colors[cell.name], marker=markers[cell.name], s=5.0
            )

            # Plot the mean and standard deviation for the targets
            for protocol in protocols:

                if protocol.name == protocol_name:

                    if key_amp == "amp_rel":
                        x = protocol.amplitude
                    elif key_amp == "amp":
                        x = numpy.mean([t.amp for t in protocol.recordings])

                    mean, std = protocol.mean_std_efeature(efeature)

                    axs[xpos][ypos].errorbar(
                        x,
                        mean,
                        yerr=std,
                        marker="o",
                        elinewidth=0.7,
                        markersize=3.0,
                        c="gray",
                        zorder=100,
                        alpha=0.6,
                    )

        if xpos is not None:
            axs[xpos][ypos].set_ylabel(efeature, size="small")

            axs[xpos][ypos].tick_params(
                axis="both", which="major", labelsize=6
            )
            axs[xpos][ypos].tick_params(
                axis="both", which="minor", labelsize=3
            )
            axs[xpos][ypos].tick_params(
                axis="both", which="major", labelsize=6
            )
            axs[xpos][ypos].tick_params(
                axis="both", which="minor", labelsize=3
            )

    # Remove surplus of subplots
    if xpos:
        for ax_idx in range(xpos + 1, PLOT_PER_COLUMN):
            fig.delaxes(axs[ax_idx, -1])

    if key_amp == "amp_rel":
        fig.text(
            0.5,
            0.01,
            r"Relative step amplitude" " ($I/I_{thresh}$)",
            ha="center",
        )
    elif key_amp == "amp":
        fig.text(0.5, 0.01, "Step amplitude (nA)", ha="center")

    fig.suptitle("Protocol: {}".format(protocol_name))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if len(cells) == 1:
        filename = "{}_{}_efeatures_{}.pdf".format(
            cells[0].name, protocol_name, key_amp
        )
        dirname = pathlib.Path(output_dir) / cells[0].name

    else:
        filename = "{}_efeatures_{}.pdf".format(protocol_name, key_amp)
        dirname = pathlib.Path(output_dir)

    _save_fig(dirname, filename)

    return fig, axs


def plot_individual_efeatures(
    cells, protocols, output_dir, colors=None, markers=None, key_amp="amp"
):
    if not colors or not markers:
        colors, markers = _get_colors_markers_wheels(cells)

    for cell in cells:

        for protocol_name in cell.get_protocol_names():

            efeatures = set()
            for c in cells:
                for rec in c.recordings:
                    if rec.protocol_name == protocol_name:
                        efeatures.update(list(rec.efeatures.keys()))

            _ = plot_efeatures(
                cells=[cell],
                protocol_name=protocol_name,
                efeatures=efeatures,
                output_dir=output_dir,
                protocols=protocols,
                key_amp=key_amp,
                colors=colors,
                markers=markers,
            )


def plot_grouped_efeatures(
    cells, protocols, output_dir, colors=None, markers=None, key_amp="amp"
):
    if not colors or not markers:
        colors, markers = _get_colors_markers_wheels(cells)

    protocol_names = set([p.name for p in protocols])

    for protocol_name in protocol_names:

        efeatures = set()
        for p in protocols:
            if p.name == protocol_name:
                efeatures.update(list(p.efeatures.keys()))

        _ = plot_efeatures(
            cells=cells,
            protocol_name=protocol_name,
            efeatures=efeatures,
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
