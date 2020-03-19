"""Igor reader for binary wave format"""

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

import re
import numpy as np
from igor import binarywave

# as we support Python 2.7/3, we have a special import for StringIO
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def _bytes_to_str(bytes_):
    """concatenates an iterable of bytes to a str

    Args:
        bytes_(numpy.ndarray): array of bytes

    Returns:
        str of concatenated bytes with omitted `x00` bytes
    """
    return ''.join(bytes_[bytes_ != b''].astype(str))


class IgorHeader(object):
    """Header metaclass information.
    """

    def __init__(self, version, content):
        header = content['wave_header']
        self.bname = header['bname'].decode('utf-8')
        self.dUnits = _bytes_to_str(header['dataUnits'])
        self.npnts = header['npnts']
        self.wavenotes = content['note'].decode('utf-8')

        if version == 5:
            self.xUnits = _bytes_to_str(header['dimUnits'])
            self.dx = header['sfA'][0]
            self.next = header['next']
            self.creationDate = header['creationDate']
            self.modDate = header['modDate']
            self.sfA = header['sfA'].astype(np.double)
            self.dimUnits = header['dimUnits'].astype(str)
            self.fsValid = header['fsValid']
            self.topFullScale = header['topFullScale']
            self.botFullScale = header['botFullScale']
            self.dataEUnits = header['dataEUnits']
            self.dimEUnits = header['dimEUnits']
            self.dimLabels = header['dimLabels']
            self.waveNoteH = header['waveNoteH']
            # self.platform = header['platform']
        else:
            self.xUnits = _bytes_to_str(header['xUnits'])
            self.dx = header['hsA']


def read_wave_notes(wavenotes):
    """parse wavenotes to collect them into a dict

    Args:
        wavenotes(str): string of wavenotes

    Returns:
        dict of wavenotes
    """
    wavenotes = dict(re.findall("(.+?):(.+?);", wavenotes))
    return wavenotes


def read_from_binary(content):
    """Reads Igor's (Wavemetric) binary wave format represented as
    `content` string. Basically it applies `read_from_handle` to `content`
    wrapped into a file handler.

    Args:
        content(str): string

    Returns:
        see `read_from_handle` output
    """
    file_handle = StringIO(content)
    return read_from_handle(file_handle)


def read(filename):
    """Reads Igor's (Wavemetric) binary wave format from a file under
    `filename` path.

    Args:
        filename(str):

    Returns:
        see `read_from_handle` output
    """

    with open(filename, "rb") as f:
        return read_from_handle(f)


def read_from_handle(f):
    """Reads Igor's (Wavemetric) binary wave format, .ibw or .bwav, files.

    Args:
        f(file): file handle

    Returns:
        A tuple of (headerType instance, numpy vector) where `headerType
        instance` contains a meta info about the wave and `numpy vector`
        contains wave data. `numpy vector` is writeable.
    """

    data = binarywave.load(f)
    version = data['version']
    assert version in (2, 3, 5), "Fileversion is '" + \
                                 str(version) + "', not supported"

    content = data['wave']
    wdata = np.copy(content['wData'])
    header = IgorHeader(version, content)
    return header, wdata
