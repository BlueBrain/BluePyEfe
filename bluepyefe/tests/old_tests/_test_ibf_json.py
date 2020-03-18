"""Test extractor functions with ibf"""

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

# pylint: disable=C0301

import os
import json

import pytest


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


def test_ibf_json(rootdir):
    """Test ibf json"""

    config_str = """
    {
      "cells": {
        "970509hp2": {
          "etype": "etype",
          "exclude": [
            [
              -1.8
            ],
            [
              -1.8
            ]
          ],
          "experiments": {
            "step": {
              "files": [
                "rattus-norvegicus____hippocampus____ca1____interneuron____cac____970509hp2____97509008",
                "rattus-norvegicus____hippocampus____ca1____interneuron____cac____970509hp2____97509009"
              ],
              "location": "soma"
            }
          },
          "ljp": 0,
          "v_corr": 0
        }
      },
      "comment": [],
      "features": {
        "step": [
          "time_to_last_spike",
          "time_to_second_spike",
          "voltage",
          "voltage_base"
        ]
      },
      "format": "ibf_json",
      "options": {
        "delay": 500,
        "logging": false,
        "nanmean": false,
        "relative": false,
        "target": [
          "all"
        ],
        "tolerance": 0.02
      },
      "path": "./data_ibf_json/eg_json_data/traces"
    }
    """

    config = json.loads(config_str)
    json.dump(config, open(os.path.join(rootdir, 'configs', 'ibf_json1.json'),
                           'w'), sort_keys=True, indent=4)
    config['path'] = os.path.join(rootdir, config['path'])

    import bluepyefe as bpefe

    extractor = bpefe.Extractor(
        'temptype_ibf', config, use_git=False)
    extractor.create_dataset()
    extractor.plt_traces()
    extractor.extract_features(threshold=-30)
    extractor.mean_features()
    extractor.plt_features()
    extractor.feature_config_cells(version='legacy')
    extractor.feature_config_all(version='legacy')
