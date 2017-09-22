config = {

    "cells":    {
                    "970509hp2" : {
                        "etype": "etype",
                        "exclude":[[-1.8],[-1.8]],
                        "experiments": {
                            "step": {
                                "files": ["rattus-norvegicus____hippocampus____ca1____interneuron____cac____970509hp2____97509008", "rattus-norvegicus____hippocampus____ca1____interneuron____cac____970509hp2____97509009"],
                                "location": "soma"
                                },
                        },
        "ljp" : 0,
        "v_corr" : 0,
        },
    },
     "comment": [],
 "features": {"step": ["time_to_last_spike", "time_to_second_spike", "voltage", "voltage_base"]},
 "format": "ibf_json",
 "options": {"delay": 500,
             "logging": False,
             "nanmean": False,
             "relative": False,
             "target": ["all"],
             "tolerance": 0.02},
 "path": "./data_ibf_json/eg_json_data/traces"
 }


import bluepyefe as bpefe

extractor = bpefe.Extractor('tempTestIbfJson', config, use_git=False)
extractor.create_dataset()
extractor.plt_traces()
extractor.extract_features(threshold=-30)
extractor.mean_features()
#extractor.analyse_threshold()
extractor.plt_features()
extractor.feature_config_cells(version='legacy')
extractor.feature_config_all(version='legacy')

