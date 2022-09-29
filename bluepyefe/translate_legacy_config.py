"""To translate config dictionaries from BluePyEfe 1 to input needed by BluePyEfe 2"""

"""
Copyright (c) 2022, EPFL/Blue Brain Project

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

import pathlib


def translate_legacy_files_metadata(config):
    """Translate the legacy field "cells" into the new files_metadata"""

    if "path" in config:
        ephys_path = pathlib.Path(config["path"])
    else:
        ephys_path = pathlib.Path("./")

    files_metadata = {}
    for cell_name in config["cells"]:
        for protocol_name in config["cells"][cell_name]["experiments"]:
            for file_metadata in config["cells"][cell_name]["experiments"][protocol_name]["files"]:

                if cell_name not in files_metadata:
                    files_metadata[cell_name] = {}
                if protocol_name not in files_metadata[cell_name]:
                    files_metadata[cell_name][protocol_name] = []

                if "i_file" in file_metadata:
                    filepaths = {
                        "i_file": str(ephys_path / file_metadata["i_file"]),
                        "v_file": str(ephys_path / file_metadata["v_file"]),
                    }
                else:
                    filepaths = {"filepath": str(ephys_path / file_metadata["filepath"])}

                files_metadata[cell_name][protocol_name].append(file_metadata)
                files_metadata[cell_name][protocol_name][-1].update(filepaths)

                if protocol_name in config["options"]["onoff"]:
                    files_metadata[cell_name][protocol_name][-1]["ton"] = config[
                        "options"]["onoff"][protocol_name][0]
                    files_metadata[cell_name][protocol_name][-1]["toff"] = config[
                        "options"]["onoff"][protocol_name][1]

    return files_metadata


def translate_legacy_targets(config):
    """Translate the legacy field "targets" into the new targets"""

    targets = []

    for protocol in config["features"]:
        for feature in config["features"][protocol]:
            if "spikerate" in feature:
                continue
            for amp, tol in zip(config["options"]["target"], config["options"]["tolerance"]):

                if amp == "all":
                    continue
                if amp == "noinput":
                    effective_amp = 0
                    effective_tolerance = 10
                else:
                    effective_amp = amp
                    effective_tolerance = tol

                efel_settings = {}
                if "strict_stiminterval" in config["options"]:
                    if protocol in config["options"]["strict_stiminterval"]:
                        efel_settings["strict_stiminterval"] = config["options"][
                            "strict_stiminterval"][protocol]
                    elif "base" in config["options"]["strict_stiminterval"]:
                        efel_settings["strict_stiminterval"] = config["options"][
                            "strict_stiminterval"]["base"]

                targets.append(
                    {
                        "efeature": feature,
                        "protocol": protocol,
                        "amplitude": effective_amp,
                        "tolerance": effective_tolerance,
                        "efel_settings": efel_settings
                    }
                )

    return targets


def translate_legacy_config(config):
    """Translate a legacy config from BluePyEfe 1 to BluePyEfe 2"""

    files_metadata = translate_legacy_files_metadata(config)
    targets = translate_legacy_targets(config)
    protocols_rheobase = config["options"]["expthreshold"]

    rheobase_strategy = "absolute"
    rheobase_settings = {}
    if "spike_threshold" in config["options"]:
        rheobase_settings["spike_threshold"] = config["options"]["spike_threshold"]

    return {
        "files_metadata": files_metadata,
        "targets": targets,
        "protocols_rheobase": protocols_rheobase,
        "rheobase_strategy": rheobase_strategy,
        "rheobase_settings": rheobase_settings
    }
