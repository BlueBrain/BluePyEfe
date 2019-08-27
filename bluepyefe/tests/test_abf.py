import os
import json

import pytest

import bluepyefe as bpefe


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


def test_abf(rootdir):
    """Test loading of abf"""

    config_str = """
    {
      "cells": {
        "cell01": {
          "etype": "etype",
          "exclude": [
            [
              -2.0
            ],
            [
              -2.0
            ]
          ],
          "experiments": {
            "step": {
              "files": [
                "96711008",
                "96711009"
              ],
              "location": "soma"
            }
          },
          "ljp": 0,
          "v_corr": 0
        },
        "cell02": {
          "etype": "etype",
          "exclude": [
            [],
            []
          ],
          "experiments": {
            "step": {
              "files": [
                "98205017",
                "98205018"
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
      "format": "axon",
      "options": {
        "delay": 500,
        "logging": false,
        "nanmean": true,
        "relative": false,
        "target": [
          -2.0,
          0.0,
          1.0
        ],
        "tolerance": 0.02
      },
      "path": "./data_abf/"
    }
    """

    config = json.loads(config_str)

    config['path'] = os.path.join(rootdir, config['path'])

    extractor = bpefe.Extractor('testtype_abf', config, use_git=False)
    extractor.create_dataset()
    extractor.plt_traces()
    extractor.extract_features(threshold=-30)
    extractor.mean_features()
    extractor.plt_features()
    extractor.feature_config_cells()
    extractor.feature_config_all()
