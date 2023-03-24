"""Tools shared by all eCodes"""

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
import numpy
from scipy.ndimage import median_filter


def scipy_signal2d(data, width):
    return median_filter(data, size=width).tolist()


def base_current(current, idx_ton=300):
    """Compute the base current from the first few points of the current
    array"""

    # Get the base current hypamp
    upper_lim = min(idx_ton, len(current))
    smooth_current = scipy_signal2d(current[:upper_lim], 85)
    return numpy.median(smooth_current)
