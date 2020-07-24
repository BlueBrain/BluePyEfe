import os
import numpy
import json


class printFeatures:
    TT_CONFIG_PATH = 'configs'
    TT_CONFIG_FILE = 'tabletools_config.json'

    @classmethod
    def dump_features(
            cls, all_feat_filename="all_features.txt", cellname="CELLNAME",
            trace_filename="", features_name=[], fel_vals=[], multvalnum=5,
            metadata={}, amp=0, stim_start=0, stim_end=0):
        counter = 0
        param_file = os.path.join(
            os.path.dirname(__file__), cls.TT_CONFIG_PATH,
            cls.TT_CONFIG_FILE)
        with open(param_file, 'r') as f:
            params = json.load(f)
        f.close()

        CELLINFO = params["CELLINFO"]
        MULTVALFEAT = params["MULTVALFEAT"]

        # create file with headers if needed
        if not os.path.exists(all_feat_filename):
            headers = []
            for i in CELLINFO:
                headers.append(i)
            headers.append('stim')
            headers.append('stim_start')
            headers.append('stim_end')
            for i in features_name:
                if i in MULTVALFEAT:
                    for j in range(multvalnum - 1):
                        headers.append(i + '___event' + str(j + 1))
                    headers.append(i + '___eventlast')
                else:
                    headers.append(i)
            headers.append('\n')
            with open(all_feat_filename, 'w') as f:
                f.write('\t'.join(headers))
        f = open(all_feat_filename, 'a')
        crr_info_str = []
        crr_sweep_str = []
        for i in CELLINFO:
            if i == 'cell_id':
                crr_info_str.append(str(cellname))
            elif i == 'filename':
                crr_info_str.append(str(trace_filename))
            else:
                crr_info_str.append(metadata[i])

        # append stim info
        crr_info_str.append(str(amp))
        crr_info_str.append(str(stim_start))
        crr_info_str.append(str(stim_end))

        # convert to string
        crr_sweep_str = list(crr_info_str)

        # for every extracted feature
        for ii in features_name:
            crr_feature = fel_vals[0][ii]
            if ii not in MULTVALFEAT:
                if crr_feature is not None and not \
                        (
                        numpy.array_equal(
                            crr_feature,
                            numpy.array([]))):
                    crr_sweep_str.append(str(crr_feature[0]))
                else:
                    crr_sweep_str.append(str(numpy.nan))
            else:
                crr_feat_val = [numpy.nan] * multvalnum
                if crr_feature is not None and len(crr_feature) > 0:
                    crr_feat_len = len(crr_feature)
                    for y in range(multvalnum - 2):
                        if len(crr_feature) > y + 1:
                            crr_feat_val[y + 1] = crr_feature[y + 1]
                    if any(crr_feature):
                        crr_feat_val[0] = crr_feature[0]
                        crr_feat_val[-1] = crr_feature[-1]

                # append values to the string to be printed
                for w in crr_feat_val:
                    crr_sweep_str.append(str(w))

        crr_final_str = '\t'.join(crr_sweep_str)
        crr_final_str += "\n"
        f.write(crr_final_str)
