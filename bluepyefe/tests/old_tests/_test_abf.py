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

    # json.dump(config, open(os.path.join(rootdir, 'configs', 'abf1.json'), 'w'), sort_keys=True, indent=4)

    config['path'] = os.path.join(rootdir, config['path'])

    extractor = bpefe.Extractor('testtype_abf', config, use_git=False)
    extractor.create_dataset()
    extractor.plt_traces()
    extractor.extract_features(threshold=-30)
    extractor.mean_features()
    extractor.plt_features()
    extractor.feature_config_cells()
    extractor.feature_config_all()
