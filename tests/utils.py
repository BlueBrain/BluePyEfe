"""Utils"""

import urllib.request
import shutil
from pathlib import Path

def download_datafiles(pathname, channels, numbers, output_dir, gb_url):
    paths = [f"{pathname}_{ch}_{n}.ibw" for ch in channels for n in numbers]

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for path in paths:
        output_path = f"{output_dir}{path}"
        if not Path(output_path).is_file():
            with urllib.request.urlopen(f"{gb_url}{path}") as response, open(output_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

def download_sahp_datafiles():
    """Download data files for sAHP and IDthresh traces."""
    output_dir = "./tests/exp_data/X/"
    gb_url = "https://raw.githubusercontent.com/BlueBrain/SSCxEModelExamples/main/feature_extraction/input-traces/C060109A1-SR-C1/"
    sahp_pathname = "X_sAHP"
    sahp_ch = ["ch0", "ch1"]
    sahp_numbers = list(range(320, 326))
    idthresh_pathname = "X_IDthresh"
    idthresh_ch = ["ch0", "ch1"]
    idthresh_numbers = list(range(349, 358)) + list(range(362, 371))

    download_datafiles(sahp_pathname, sahp_ch, sahp_numbers, output_dir, gb_url)
    download_datafiles(idthresh_pathname, idthresh_ch, idthresh_numbers, output_dir, gb_url)

def download_apthresh_datafiles():
    """Download data files for APThreshold and IDthresh traces."""
    output_dir = "./tests/exp_data/X/"
    gb_url = "https://raw.githubusercontent.com/BlueBrain/SSCxEModelExamples/main/feature_extraction/input-traces/C060109A1-SR-C1/"
    apthresh_pathname = "X_APThreshold"
    apthresh_ch = ["ch0", "ch1"]
    apthresh_numbers = list(range(254, 257))
    idthresh_pathname = "X_IDthresh"
    idthresh_ch = ["ch0", "ch1"]
    idthresh_numbers = list(range(349, 358)) + list(range(362, 371))

    download_datafiles(apthresh_pathname, apthresh_ch, apthresh_numbers, output_dir, gb_url)
    download_datafiles(idthresh_pathname, idthresh_ch, idthresh_numbers, output_dir, gb_url)
