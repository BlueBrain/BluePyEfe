import os
import json
import glob

import pytest

import bluepyefe as bpefe


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


def test_config(rootdir):
    """Test config"""

    configs_dir = os.path.join(rootdir, 'configs')

    config_paths = glob.glob(os.path.join(configs_dir, '*.json'))

    for config_path in config_paths:
        config = json.load(open(config_path))

        config['path'] = os.path.join(rootdir, config['path'])

        '''
        extractor = bpefe.Extractor('testtype_legacy', config, use_git=False)
        extractor.create_dataset()
        extractor.plt_traces()
        extractor.extract_features(threshold=-30)
        extractor.mean_features()
        extractor.analyse_threshold()
        extractor.plt_features()
        extractor.feature_config_cells(version='legacy')
        extractor.feature_config_all(version='legacy')
        '''

        extractor = bpefe.Extractor('testtype', config, use_git=False)
        extractor.create_dataset()
        extractor.plt_traces()
        extractor.extract_features(threshold=-30)
        extractor.mean_features()
        # extractor.analyse_threshold()
        extractor.plt_features()
        extractor.feature_config_cells()
        extractor.feature_config_all()
        extractor.plt_features_dist()
