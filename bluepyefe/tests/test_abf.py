config = {

    "cells":    {
                    "cell01" : {
                        "etype": "etype",
                        "exclude":[[-1.8],[-1.8]],
                        "experiments": {
                            "step": {
                                "files": ["96711008", "96711009"],
                                "location": "soma"
                                },
                        },
        "ljp" : 0,
        "v_corr" : 0,
        },
        "cell02" : {
                        "etype": "etype",
                        "exclude":[[-1.8],[-1.8]],
                        "experiments": {
                            "step": {
                                "files": ["98205017", "98205018"],
                                "location": "soma"
                                },
                        },
        "ljp" : 0,
        "v_corr" : 0,
        },
    },
     "comment": [],
 "features": {"step": ["time_to_last_spike", "time_to_second_spike", "voltage", "voltage_base"]},
 "format": "axon",
 "options": {"delay": 500,
             "logging": False,
             "nanmean": False,
             "relative": False,
             "target": ["all"],
             "tolerance": 0.02},
 "path": "./data_abf/"
 }


import bluepyefe as bpefe

extractor = bpefe.Extractor('testABF', config, use_git=False)
extractor.create_dataset()
extractor.plt_traces()
extractor.extract_features(threshold=-30)
extractor.mean_features()
#extractor.analyse_threshold()
extractor.plt_features()
extractor.feature_config_cells(version='legacy')
extractor.feature_config_all(version='legacy')

