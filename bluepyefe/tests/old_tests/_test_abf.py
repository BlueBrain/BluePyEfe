"""Test extractor functions with abf"""

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

import os
import json

import pytest

import bluepyefe as bpefe


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


def test_abf(rootdir):
    """Test loading of abf"""

    config = json.load(open(os.path.join(rootdir, 'configs', 'abf1.json')))
    config['path'] = os.path.join(rootdir, config['path'])

    extractor = bpefe.Extractor('testtype_abf', config)
    extractor.create_dataset()
    extractor.plt_traces()
    extractor.extract_features(threshold=-30)
    extractor.mean_features()
    extractor.plt_features()
    extractor.feature_config_cells()
    extractor.feature_config_all()
