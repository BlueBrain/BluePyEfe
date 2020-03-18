"""Test extractor functions"""

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


def test_config(rootdir):
    """Test config"""

    config_str = """
    {
      "comment": [
        "v_corr: normalize membrane potential to this value",
        "ljp: set so to 14mV",
        "etype: was defined individually by eye from plotted traces"
      ],
      "features": {
        "IDthresh": [
          "adaptation_index2",
          "mean_frequency",
          "time_to_first_spike",
          "ISI_log_slope",
          "ISI_log_slope_skip",
          "time_to_last_spike",
          "inv_time_to_first_spike",
          "inv_first_ISI",
          "inv_second_ISI",
          "inv_third_ISI",
          "inv_fourth_ISI",
          "inv_fifth_ISI",
          "inv_last_ISI",
          "voltage_deflection",
          "voltage_deflection_begin",
          "steady_state_voltage",
          "decay_time_constant_after_stim",
          "AP_amplitude"
        ]
      },
      "format": "igor",
      "cells": {
        "C060109A2-SR-C1": {
          "ljp": 14,
          "v_corr": false,
          "experiments": {
            "IDthresh": {
              "files": [
                {
                  "ordinal": "348",
                  "i_unit": "A",
                  "v_file": "C060109A2-SR-C1/X_IDthresh_ch1_348.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A2-SR-C1/X_IDthresh_ch0_348.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "349",
                  "i_unit": "A",
                  "v_file": "C060109A2-SR-C1/X_IDthresh_ch1_349.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A2-SR-C1/X_IDthresh_ch0_349.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "350",
                  "i_unit": "A",
                  "v_file": "C060109A2-SR-C1/X_IDthresh_ch1_350.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A2-SR-C1/X_IDthresh_ch0_350.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "351",
                  "i_unit": "A",
                  "v_file": "C060109A2-SR-C1/X_IDthresh_ch1_351.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A2-SR-C1/X_IDthresh_ch0_351.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "352",
                  "i_unit": "A",
                  "v_file": "C060109A2-SR-C1/X_IDthresh_ch1_352.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A2-SR-C1/X_IDthresh_ch0_352.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                }
              ],
              "location": "soma.v"
            }
          }
        },
        "C060109A1-SR-C1": {
          "ljp": 14,
          "v_corr": false,
          "experiments": {
            "IDthresh": {
              "files": [
                {
                  "ordinal": "349",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_349.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_349.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "350",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_350.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_350.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "351",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_351.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_351.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "352",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_352.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_352.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "353",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_353.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_353.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "354",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_354.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_354.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "355",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_355.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_355.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "356",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_356.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_356.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "357",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_357.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_357.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "362",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_362.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_362.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "363",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_363.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_363.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "364",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_364.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_364.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "365",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_365.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_365.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "366",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_366.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_366.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "367",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_367.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_367.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "368",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_368.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_368.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "369",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_369.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_369.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                },
                {
                  "ordinal": "370",
                  "i_unit": "A",
                  "v_file": "C060109A1-SR-C1/X_IDthresh_ch1_370.ibw",
                  "t_unit": "s",
                  "i_file": "C060109A1-SR-C1/X_IDthresh_ch0_370.ibw",
                  "v_unit": "V",
                  "dt": 0.00025
                }
              ],
              "location": "soma.v"
            }
          }
        }
      },
      "path": "./data/",
      "options": {
        "expthreshold": [
          "IDrest",
          "IDthresh"
        ],
        "relative": true,
        "delay": 0,
        "nanmean": false,
        "target": [
          -20,
          "noinput",
          50,
          100,
          120,
          "all"
        ],
        "tolerance": [
          5,
          false,
          5,
          5,
          5,
          false
        ],
        "nangrace": 0,
        "spike_threshold": 1,
        "amp_min": 0,
        "strict_stiminterval": {
          "SpikeRec": false,
          "base": true
        },
        "onoff": {
          "TesteCode": [
            100.0,
            600.0
          ],
          "APWaveform": [
            5.0,
            55.0
          ],
          "IV": [
            20.0,
            1020.0
          ],
          "IDrest": [
            700.0,
            2700.0
          ],
          "SpontAPs": [
            100.0,
            10100.0
          ],
          "APDrop": [
            10.0,
            15.0
          ],
          "IRhyperpol": [
            500.0,
            700.0
          ],
          "Spontaneous": [
            100.0,
            10100.0
          ],
          "IDthresh": [
            700.0,
            2700.0
          ],
          "APThreshold": [
            0.0,
            2000.0
          ],
          "SpikeRec": [
            10.0,
            13.5
          ],
          "IDdepol": [
            700.0,
            2700.0
          ],
          "Step": [
            700.0,
            2700.0
          ],
          "Delta": [
            10.0,
            60.0
          ],
          "IRdepol": [
            500.0,
            700.0
          ]
        },
        "logging": true,
        "boxcox": false
      }
    }
    """

    config = json.loads(config_str)
    json.dump(config, open(os.path.join(rootdir, 'configs', 'ibw1.json'), 'w'),
              sort_keys=True, indent=4)

    config['path'] = os.path.join(rootdir, config['path'])

    extractor = bpefe.Extractor('testtype_legacy', config, use_git=False)
    extractor.create_dataset()
    extractor.plt_traces()
    extractor.extract_features(threshold=-30)
    extractor.mean_features()
    extractor.analyse_threshold()
    extractor.plt_features()
    extractor.feature_config_cells(version='legacy')
    extractor.feature_config_all(version='legacy')

    extractor = bpefe.Extractor('testtype', config, use_git=False)
    extractor.create_dataset()
    extractor.plt_traces()
    extractor.extract_features(threshold=-30)
    extractor.mean_features()
    extractor.analyse_threshold()
    extractor.plt_features()
    extractor.feature_config_cells()
    extractor.feature_config_all()
    extractor.plt_features_dist()
