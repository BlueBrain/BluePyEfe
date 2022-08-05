"""Plotting tools"""

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


import matplotlib
import matplotlib.colors as mplcol
import matplotlib.pyplot as plt
import colorsys
import numpy
import collections


def adjust_spines(ax, spines, color='k', d_out=10, d_down=[]):

    if d_down == []:
        d_down = d_out

    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    for loc, spine in ax.spines.items():
        if loc in spines:
            if loc == 'bottom':
                spine.set_position(('outward', d_down))  # outward by 10 points
            else:
                spine.set_position(('outward', d_out))  # outward by 10 points
            # spine.set_smart_bounds(True)
        else:
            spine.set_visible(False)  # set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')

        if color != 'k':

            ax.spines['left'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)

    elif 'right' not in spines:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'right' in spines:
        ax.yaxis.set_ticks_position('right')

        if color != 'k':

            ax.spines['right'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)

    if 'bottom' in spines:
        pass
        ax.xaxis.set_ticks_position('bottom')
        # ax.axes.get_xaxis().set_visible(True)

    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        ax.axes.get_xaxis().set_visible(False)


def light_palette(color, n_colors=6, reverse=False, lumlight=0.8, light=None):

    rgb = mplcol.colorConverter.to_rgb(color)

    if light is not None:
        light = mplcol.colorConverter.to_rgb(light)
    else:
        vals = list(colorsys.rgb_to_hls(*rgb))
        vals[1] = lumlight
        light = colorsys.hls_to_rgb(*vals)

    colors = [color, light] if reverse else [light, color]
    pal = mplcol.LinearSegmentedColormap.from_list("blend", colors)

    palette = pal(numpy.linspace(0, 1, n_colors))

    return palette


def tiled_figure(
    figname='', frames=1, columns=2, rows_per_page=4,
    dirname='', figs=collections.OrderedDict(), orientation='page',
    width_ratios=None, height_ratios=None, hspace=0.6, wspace=0.2,
    top=0.97, bottom=0.05, left=0.07, right=0.97
):

    if figname not in figs.keys():

        if orientation == 'landscape':
            figsize = (297 / 25.4, 210 / 25.4)
        elif orientation == 'page':
            figsize = (210 / 25.4, 297 / 25.4)

        params = {
            'backend': 'ps',
            'axes.labelsize': 6,
            'axes.linewidth': 0.5,
            'font.size': 8,
            'axes.titlesize': 8,
            'legend.fontsize': 8,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.borderpad': 0.2,
            'legend.loc': 'best',
            'text.usetex': False,
            # 'pdf.fonttype': 42,
            'figure.figsize': figsize
        }
        matplotlib.rcParams.update(params)

        axs = []
        figs[figname] = {}
        figs[figname]['dirname'] = dirname
        figs[figname]['fig'] = []

        if frames < columns:
            rows_per_page = 1
            columns = frames
        elif frames < (columns * rows_per_page):
            rows_per_page = numpy.ceil(frames / float(columns)).astype(int)

        if width_ratios is None:
            width_ratios = [1] * columns

        if height_ratios is None:
            height_ratios = [1] * rows_per_page

        gs = matplotlib.gridspec.GridSpec(
            rows_per_page, columns,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            hspace=hspace,
            wspace=wspace
        )

        """
        Compute the number of frames per page and the number of pages.
        Then define the figures, one per page
        """
        frames_per_page = rows_per_page * columns
        pages = numpy.ceil(frames / float(frames_per_page)).astype(int)
        for page in range(pages):
            fig = plt.figure(figname + "_" + str(page), facecolor='white')
            figs[figname]['fig'].append(fig)
            for i_frame in range(frames_per_page):
                axs.append(fig.add_subplot(
                    gs[int(i_frame / columns), int(i_frame % columns)]
                ))
                adjust_spines(axs[-1], ['left', 'bottom'], d_out=0)

        figs[figname]['axs'] = axs

    else:
        axs = figs[figname]['axs']

    return axs
