import logging
import numpy

logger = logging.getLogger(__name__)


class NWBReader:
    def __init__(self, content, target_protocols, repetition=None, v_file=None):
        """ Init

        Args:
            content (h5.File): NWB file
            target_protocols (list of str): list of the protocols to be read and returned
            repetition (list of int): id of the repetition(s) to be read and returned
            v_file (str): name of original file that can be retrieved in sweep's description
        """

        self.content = content
        self.target_protocols = target_protocols
        self.repetition = repetition
        self.v_file = v_file

    def read(self):
        """ Read the content of the NWB file

        Returns:
            data (list of dict): list of traces"""

        raise NotImplementedError()

    def _format_nwb_trace(self, voltage, current, start_time, trace_name=None, repetition=None):
        """ Format the data from the NWB file to the format used by BluePyEfe

        Args:
            voltage (Dataset): voltage series
            current (Dataset): current series
            start_time (Dataset): starting time
            trace_name (Dataset): name of the trace
            repetition (int): repetition number

        Returns:
            dict: formatted trace
        """

        v_array = numpy.array(
            voltage[()] * voltage.attrs["conversion"], dtype="float32"
        )

        i_array = numpy.array(
            current[()] * current.attrs["conversion"], dtype="float32"
        )

        dt = 1. / float(start_time.attrs["rate"])

        v_unit = voltage.attrs["unit"]
        i_unit = current.attrs["unit"]
        t_unit = start_time.attrs["unit"]
        if not isinstance(v_unit, str):
            v_unit = voltage.attrs["unit"].decode('UTF-8')
            i_unit = current.attrs["unit"].decode('UTF-8')
            t_unit = start_time.attrs["unit"].decode('UTF-8')

        return {
            "voltage": v_array,
            "current": i_array,
            "dt": dt,
            "id": str(trace_name),
            "repetition": repetition,
            "i_unit": i_unit,
            "v_unit": v_unit,
            "t_unit": t_unit,
        }


class AIBSNWBReader(NWBReader):
    def read(self):
        """ Read the content of the NWB file

        Returns:
            data (list of dict): list of traces"""

        data = []

        for sweep in list(self.content["acquisition"]["timeseries"].keys()):
            protocol_name = self.content["acquisition"]["timeseries"][sweep]["aibs_stimulus_name"][()]
            if not isinstance(protocol_name, str):
                protocol_name = protocol_name.decode('UTF-8')

            if (
                self.target_protocols and
                protocol_name.lower() not in [prot.lower() for prot in self.target_protocols]
            ):
                continue

            data.append(self._format_nwb_trace(
                voltage=self.content["acquisition"]["timeseries"][sweep]["data"],
                current=self.content["stimulus"]["presentation"][sweep]["data"],
                start_time=self.content["acquisition"]["timeseries"][sweep]["starting_time"],
                trace_name=sweep
            ))

        return data


class ScalaNWBReader(NWBReader):
    def read(self):
        """ Read and format the content of the NWB file

        Returns:
            data (list of dict): list of traces
        """

        data = []

        for sweep in list(self.content['acquisition'].keys()):
            key_current = sweep.replace('Series', 'StimulusSeries')
            try:
                protocol_name = self.content["acquisition"][sweep].attrs["stimulus_description"]
            except KeyError:
                logger.warning(f'Could not find "stimulus_description" attribute for {sweep}, Setting it as "Step"')
                protocol_name = "Step"

            if ("na" == protocol_name.lower()) or ("step" in protocol_name.lower()):
                protocol_name = "Step"

            if (
                self.target_protocols and
                protocol_name.lower() not in [prot.lower() for prot in self.target_protocols]
            ):
                continue

            if key_current not in self.content['stimulus']['presentation']:
                continue

            data.append(self._format_nwb_trace(
                voltage=self.content['acquisition'][sweep]['data'],
                current=self.content['stimulus']['presentation'][key_current]['data'],
                start_time=self.content["acquisition"][sweep]["starting_time"],
                trace_name=sweep,
            ))

        return data


class BBPNWBReader(NWBReader):
    def _get_repetition_keys_nwb(self, ecode_content, request_repetitions=None):
        """ Filter the names of the traces based on the requested repetitions

        Args:
            ecode_content (dict): content of the NWB file for one eCode/protocol
            request_repetitions (list of int): identifier of the requested repetitions

        Returns:
            list of str: list of the keys of the traces to be read
        """

        if isinstance(request_repetitions, (int, str)):
            request_repetitions = [int(request_repetitions)]

        reps = list(ecode_content.keys())
        reps_id = [int(rep.replace("repetition ", "")) for rep in reps]

        if request_repetitions:
            return [reps[reps_id.index(i)] for i in request_repetitions]
        else:
            return list(ecode_content.keys())

    def read(self):
        """ Read and format the content of the NWB file

        Returns:
            data (list of dict): list of traces
        """

        data = []

        for ecode in self.target_protocols:
            for cell_id in self.content["data_organization"].keys():
                if ecode not in self.content["data_organization"][cell_id]:
                    new_ecode = next(
                        iter(
                            ec
                            for ec in self.content["data_organization"][cell_id]
                            if ec.lower() == ecode.lower()
                        ),
                        None
                    )
                    if new_ecode:
                        logger.debug(
                            f"Could not find {ecode} in nwb file, will use {new_ecode} instead"
                        )
                        ecode = new_ecode
                    else:
                        logger.debug(f"No eCode {ecode} in nwb.")
                        continue

                ecode_content = self.content["data_organization"][cell_id][ecode]

                rep_iter = self._get_repetition_keys_nwb(
                    ecode_content, request_repetitions=self.repetition
                )

                for rep in rep_iter:
                    for sweep in ecode_content[rep].keys():
                        for trace_name in list(ecode_content[rep][sweep].keys()):
                            if "ccs_" in trace_name:
                                key_current = trace_name.replace("ccs_", "ccss_")
                            elif "ic_" in trace_name:
                                key_current = trace_name.replace("ic_", "ics_")
                            else:
                                continue

                            if key_current not in self.content["stimulus"]["presentation"]:
                                logger.debug(f"Ignoring {key_current} not"
                                             " present in the stimulus presentation")
                                continue

                            if trace_name not in self.content["acquisition"]:
                                logger.debug(f"Ignoring {trace_name} not"
                                             " present in the acquisition")
                                continue

                            # if we have v_file, check that trace comes from this original file
                            if self.v_file is not None:
                                attrs = self.content["acquisition"][trace_name].attrs
                                if "description" not in attrs:
                                    logger.warning(
                                        "Ignoring %s because no description could be found.",
                                        trace_name
                                    )
                                    continue
                                v_file_end = self.v_file.split("/")[-1]
                                if v_file_end != attrs.get("description", "").split("/")[-1]:
                                    logger.debug(f"Ignoring {trace_name} not matching v_file")
                                    continue

                            data.append(self._format_nwb_trace(
                                voltage=self.content["acquisition"][trace_name]["data"],
                                current=self.content["stimulus"]["presentation"][key_current][
                                    "data"],
                                start_time=self.content["stimulus"]["presentation"][key_current][
                                    "starting_time"],
                                trace_name=trace_name,
                                repetition=int(rep.replace("repetition ", ""))
                            ))

        return data
