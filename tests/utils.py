"""Utils"""

import urllib.request
import shutil
from pathlib import Path


def download_sahp_datafiles():
    """Download data files for sAHP and IDthresh traces"""
    output_dir = "./tests/exp_data/X/"
    gb_url = "https://raw.githubusercontent.com/BlueBrain/SSCxEModelExamples/main/feature_extraction/input-traces/C060109A1-SR-C1/"
    sahp_pathname = "X_sAHP"
    sahp_ch = ["ch0", "ch1"]
    sahp_numbers = list(range(320, 326))
    idthresh_pathname = "X_IDthresh"
    idthresh_ch = ["ch0", "ch1"]
    idthresh_numbers = list(range(349, 358)) + list(range(362, 371))
    # https://raw.githubusercontent.com/BlueBrain/SSCxEModelExamples/tree/main/feature_extraction/input-traces/C060109A1-SR-C1/
    # https://github.com/BlueBrain/SSCxEModelExamples/blob/c24f096495150698547741b9a34dab84e0335649/feature_extraction/input-traces/C060109A1-SR-C1/X_IDthresh_ch1_370.ibw
    # https://github.com/BlueBrain/SSCxEModelExamples/raw/main/feature_extraction/input-traces/C060109A1-SR-C1/X_IDthresh_ch1_370.ibw
    # https://raw.githubusercontent.com/BlueBrain/SSCxEModelExamples/main/feature_extraction/input-traces/C060109A1-SR-C1//X_IDthresh_ch1_370.ibw

    sahp_paths = [f"{sahp_pathname}_{ch}_{n}.ibw" for ch in sahp_ch for n in sahp_numbers]
    idthresh_paths = [f"{idthresh_pathname}_{ch}_{n}.ibw" for ch in idthresh_ch for n in idthresh_numbers]
    pathnames = sahp_paths + idthresh_paths

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for pathname in pathnames:
        output_path = f"{output_dir}{pathname}"
        if not Path(output_path).is_file():
            with urllib.request.urlopen(f"{gb_url}{pathname}") as response, open(output_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)