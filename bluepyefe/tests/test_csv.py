import os
import json

import pytest


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


def test_csv(rootdir):
    """Test loading from csv"""

    config_str = """
        {
      "features": {
        "step": [
          "ISI_log_slope",
          "mean_frequency",
          "adaptation_index2",
          "ISI_CV",
          "AP_height",
          "AHP_depth_abs",
          "AHP_depth_abs_slow",
          "AHP_slow_time",
          "AP_width",
          "AP_amplitude",
          "AP1_amp",
          "AP2_amp",
          "APlast_amp",
          "AP_duration_half_width",
          "AHP_depth",
          "fast_AHP",
          "AHP_time_from_peak",
          "voltage_deflection",
          "voltage_deflection_begin",
          "voltage_base",
          "steady_state_voltage",
          "Spikecount",
          "time_to_last_spike",
          "time_to_first_spike",
          "inv_time_to_first_spike",
          "inv_first_ISI",
          "inv_second_ISI",
          "inv_third_ISI",
          "inv_fourth_ISI",
          "inv_fifth_ISI",
          "inv_last_ISI",
          "decay_time_constant_after_stim",
          "AP_begin_voltage",
          "AP_rise_time",
          "AP_fall_time",
          "AP_rise_rate",
          "AP_fall_rate"
        ]
      },
      "path": "./data_csv/",
      "format": "csv_lccr",
      "comment": [
        "cells named using name of first trace file belonging to this cell",
        "v_corr: normalize membrane potential to this value (given in UCL excel sheet)",
        "ljp: set so that RMP (UCL) - 10mV ljp = RMP (Golding 2001) - 14mV ljp",
        "etype: was defined individually by eye from plotted traces"
      ],
      "cells": {
        "TEST_CELL": {
          "v_corr": false,
          "ljp": 14.4,
          "experiments": {
            "step": {
              "location": "soma",
              "files": [
                "s150420-0403_ch1_cols",
                "s150420-0404_ch1_cols"
              ],
              "dt": 0.2,
              "startstop": [
                200,
                1000
              ],
              "amplitudes": [
                0.01,
                -0.01,
                0.02,
                -0.02,
                0.03,
                -0.03,
                0.04,
                -0.04,
                0.05,
                -0.05,
                0.06,
                -0.06,
                0.07,
                -0.07,
                0.08,
                -0.08,
                0.09,
                -0.09,
                0.1,
                -0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6
              ],
              "hypamp": 0.0,
              "ton": 200,
              "toff": 1000
            }
          }
        }
      },
      "options": {
        "relative": false,
        "tolerance": 0.01,
        "target": [
          0.02,
          -0.02,
          0.04,
          -0.04,
          0.06,
          -0.06,
          0.08,
          -0.08,
          0.1,
          -0.1,
          0.2,
          0.3,
          0.4,
          0.5,
          0.6
        ],
        "delay": 200,
        "nanmean": false
      }
    }
    """

    config = json.loads(config_str)
    config['path'] = os.path.join(rootdir, config['path'])

    import bluepyefe as bpefe

    extractor = bpefe.Extractor('testtype_csv', config, use_git=False)
    extractor.create_dataset()
    extractor.plt_traces()
    extractor.extract_features()
    extractor.mean_features()
    # extractor.analyse_threshold()
    extractor.feature_config_cells()
    extractor.feature_config_all()
    extractor.plt_features()
