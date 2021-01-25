from neo import io
from collections import OrderedDict
import os
import json
import hashlib
import calendar
import re
import six
import logging
logger = logging.getLogger(__name__)
import quantities as pq


class manageFiles:
    @classmethod
    def md5(cls, filename):
        '''
        Generate hash md5 code for the filename passed as parameter
        '''
        hash_md5 = hashlib.md5()

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class manageMetadata:
    """
    Class to read and manage information from metadata
    """

    # pattern for string substitution
    pattern = re.compile(r"[ \\.,\/]")

    @classmethod
    def get_cell_info(cls, filename_meta):
        """
        Extract cell info from the metadata file
        """

        try:
            # if a .json metadata file exists
            with open(filename_meta) as f:
                data = json.load(f)

                if "animal_species" not in data or \
                        data["animal_species"] is None or \
                        data["animal_species"].lower() == "unknown":
                    c_species = "unknown_species"
                else:
                    c_species = data["animal_species"]
                    c_species = cls.pattern.sub("-", c_species)
                    # c_species = c_species.replace(" ", "-")

                if "brain_structure" not in data or \
                        data["brain_structure"] is None or \
                        data["brain_structure"].lower() == "unknown":
                    c_area = "unknown_area"
                else:
                    c_area = data["brain_structure"]
                    c_area = cls.pattern.sub("-", c_area)

                if "cell_soma_location" not in data or \
                        data["cell_soma_location"] is None or \
                        data["cell_soma_location"].lower() == "unknown":
                    c_region = "unknown_region"
                else:
                    c_region = data["cell_soma_location"]
                    c_region = cls.pattern.sub("-", c_region)

                if "cell_type" not in data or data["cell_type"] is None or \
                        data["cell_type"].lower() == "unknown":
                    c_type = "unknown_type"
                else:
                    c_type = data["cell_type"]
                    c_type = cls.pattern.sub("-", c_type)

                if "etype" not in data or data["etype"] is None or \
                        data["etype"] == "unknown":
                    c_etype = "unknown_etype"
                else:
                    c_etype = data["etype"]
                    c_etype = cls.pattern.sub("-", c_etype)

                if "contributors_affiliations" not in data or \
                        data["contributors_affiliations"] is None or \
                        data["contributors_affiliations"].lower() == "unknown":
                    c_contrib = "unknown_contrib"
                else:
                    c_contrib = data["contributors_affiliations"]
                    c_contrib = cls.pattern.sub("-", c_contrib)

                if "cell_id" not in data or data["cell_id"] is None or \
                        data["cell_id"] == "unknown":
                    head, c_name = \
                        os.path.split(os.path.split(filename_meta)[0])
                else:
                    c_name = data["cell_id"]
                    c_name = cls.pattern.sub("-", c_name)

                if "filename" not in data or data["filename"] is None or \
                        data["filename"] == "unknown":
                    base = os.path.basename(os.path.normpath(filename_meta))
                    c_sample = os.path.splitext(base)[0]

                else:
                    c_sample = data["filename"]
                    c_sample = os.path.splitext(c_sample)[0]
                    c_sample = cls.pattern.sub("-", c_sample)

        except Exception as e:
            # if there is no metadata
            c_area = "unknown_area"
            c_species = "unknown_species"
            c_region = "unknown_region"
            c_type = "unknown_type"
            c_etype = "unknown_etype"
            c_contrib = "unknown_contributors"

            abf_meta_fn = cls.get_abf_filename(filename_meta)
            base = os.path.basename(os.path.normpath(abf_meta_fn))
            head, c_name = os.path.split(os.path.split(abf_meta_fn)[0])
            c_sample = os.path.splitext(base)[0]

        return (c_species, c_area, c_region, c_type, c_etype,
                c_contrib, c_name, c_sample)

    @classmethod
    def get_metadata(cls, metadata_file):
        """
        Read metadata file into a json dictionary
        """
        logger.info("Reading metadata " + os.path.basename(metadata_file))

        crr_file_dir = os.path.dirname(__file__)
        default_metadata_file = os.path.join(
            crr_file_dir, 'configs', 'metadata_template.json')
        with open(default_metadata_file, 'r') as mf:
            metadata = json.load(mf)
        mf.close()

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except Exception as e:
                pass

        return metadata

    @classmethod
    def get_metadata_filename(cls, filename):
        """
        Build metadata filename, based on data filename
        """
        filepath, name = os.path.split(filename)
        name_no_ext, extension = os.path.splitext(name)
        metadata_file = os.path.join(filepath, name_no_ext + '_metadata.json')

        return metadata_file

    @classmethod
    def get_abf_filename(cls, filename_meta):
        """
        Build data filename based on metadata filename
        """

        filepath, name = os.path.split(filename_meta)
        name_no_ext, extension = os.path.splitext(name)
        newname = name_no_ext.replace('_metadata', '')
        abf_file = os.path.join(filepath, newname + '.abf')

        return abf_file

    @classmethod
    def extract_authorized_collab(cls, metadata_file):
        """
        Extract authorized collab from metadata file
        """

        with open(metadata_file) as meta:
            all_meta = json.load(meta)

        return all_meta['authorized_collabs']

    @classmethod
    def generate_json_data(cls, obj, outfilename, json_dir):
        """
        Dump json data to file
        """
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        outfilepath = os.path.join(json_dir, outfilename)

        if not os.path.isfile(outfilepath):
            with open(outfilepath, 'w') as f:
                json.dump(obj, f)
        return

    @classmethod
    def generate_authorization_json(
            cls, final_json_name_list, metadatalist, final_dir):

        files_authorization = {}
        crr_file_auth_collab = extract_authorized_collab(metadatalist)
        files_authorization[final_json_name_list] = crr_file_auth_collab

        file_auth_fullpath = os.path.join(
            final_dir, "files_authorization.json")
        with open(file_auth_fullpath, 'a') as fa:
            json.dump(files_authorization, fa)

        return True

    @classmethod
    def stim_feats_from_meta(cls, crr_dict, num_segments):

        # read stimulus information
        try:
            if 'filename' not in crr_dict or not crr_dict['filename'] or \
                    crr_dict["filename"] == "unknown":
                raise Exception("'filename' key absent in metadata")

            if 'stimulus_type' not in crr_dict or \
                    crr_dict['stimulus_type'] != "step" or \
                    crr_dict["stimulus_type"] == "unknown":
                raise Exception("'stimulus_type' key absent in metadata \
                        for file:" + crr_dict['filename'])
            else:
                ty = str(crr_dict['stimulus_type'])
                logger.info("extracted stimulus type")

            if 'stimulus_time_unit' not in crr_dict or not \
                    crr_dict['stimulus_time_unit'] or \
                    crr_dict["stimulus_time_unit"] == "unknown":
                raise Exception("'stimulus_time_unit' key absent in metadata \
                    for file:" + crr_dict['filename'])
            else:
                tu = crr_dict['stimulus_time_unit']
                logger.info("extracted stimulus unit")

            if 'stimulus_start' in crr_dict \
                    and crr_dict['stimulus_start'] and \
                    crr_dict["stimulus_start"] != "unknown" and \
                    'stimulus_end' in crr_dict and \
                    crr_dict['stimulus_end'] and \
                    crr_dict['stimulus_end'] != 'uknown':
                st = crr_dict['stimulus_start'][0]
                en = crr_dict['stimulus_end'][0]
                logger.info("extracted stimulus start and stimulus \
                        end")
            elif 'tamp' in crr_dict and crr_dict['tamp']:
                st = crr_dict['tamp'][0]
                en = crr_dict['tamp'][1]
                logger.info("extracted stimulus start")
            elif 'ton' in crr_dict and 'toff' in crr_dict:
                st = crr_dict['ton']
                en = crr_dict['toff']
                logger.info("extracted stimulus start")
            else:
                raise Exception("'stimulus_start' and/or 'stimulus_end' key \
                        absent in metadata, for file:" + crr_dict['filename'])

            if 'stimulus_unit' in crr_dict and crr_dict['stimulus_unit'] and \
                    crr_dict["stimulus_unit"] != "unknown":
                u = str(crr_dict['stimulus_unit'])
                logger.info("extracted stimulus unit")
            elif 'i_unit' in crr_dict and crr_dict['i_unit'] and \
                    crr_dict['i_unit'] != 'unknown':
                u = str(crr_dict['i_unit'])
                logger.info("extracted stimulus unit")
            else:
                raise Exception("'stimulus_unit' key absent in metadata, \
                        for file:" + crr_dict['filename'])

            if 'stimulus_first_amplitude' not in crr_dict or not \
                    crr_dict['stimulus_first_amplitude'] or \
                    crr_dict["stimulus_first_amplitude"] == "unknown":
                raise Exception("'stimulus_first_amplitude' key absent in \
                        metadata, for file:" + crr_dict['filename'])
            else:
                fa = float(
                    format(
                        crr_dict['stimulus_first_amplitude'][0],
                        '.3f'))
                logger.info("extracted stimulus first amplitude")

            if 'stimulus_increment' not in crr_dict \
                    or crr_dict["stimulus_increment"] == "unknown":
                raise Exception("'stimulus_increment' key absent in metadata, \
                        for file:" + crr_dict['filename'])
            elif not crr_dict['stimulus_increment']:
                inc = float(format(0, '.3f'))
            else:
                inc = float(format(crr_dict['stimulus_increment'][0], '.3f'))
            logger.info("extracted stimulus increment")

            if 'sampling_rate_unit' not in crr_dict or not \
                    crr_dict['sampling_rate_unit'] \
                    or crr_dict["sampling_rate_unit"] == "unknown":
                ru = 'Hz'
                logger.info("extracted sampling rate unit")
            else:
                ru = crr_dict['sampling_rate_unit'][0]
                logger.info("extracted sampling rate unit")

            if 'sampling_rate' not in crr_dict or not \
                    crr_dict['sampling_rate'] or \
                    crr_dict["sampling_rate"] == "unknown":
                raise Exception("sampling_rate key absent in metadata, for \
                        file:" + crr_dict['filename'])
            else:
                r = crr_dict['sampling_rate'][0]
                logger.info("extracted sampling rate")

            if tu == 's':
                st = st * 1e3
                en = en * 1e3
        except Exception as e:
            logger.error(
                'Error in reading keys for stimulus extraction in file: ' +
                crr_dict['filename']
            )
            logger.error('ERROR ' + str(e))

        all_stim_feats = {
            "ty": [],
            "st": [],
            "en": [],
            "crr_val": [],
            "u": [],
            "ru": [],
            "r": []
        }

        if not fa:
            fa = 0

        # for every segment in the data file, compute stimulus amplitude
        for i in range(num_segments):
            crr_val = float(format(fa + inc * float(format(i, '.3f')), '.3f'))
            all_stim_feats["ty"].append(ty)
            all_stim_feats["st"].append(st)
            all_stim_feats["en"].append(en)
            all_stim_feats["crr_val"].append(crr_val)
            all_stim_feats["u"].append(u)
            all_stim_feats["r"].append(r)
            all_stim_feats["ru"].append(ru)

        return (1, all_stim_feats)

    @classmethod
    def generate_citation_json(cls, jsonfilenamelist, metadatalist, final_dir):
        """
        Generate a file with 'How to cite' information
        """
        citation_json = {}
        crrjsonfile = jsonfilenamelist
        with open(metadatalist) as cf:
            cmd = json.load(cf)
            caff = cmd['contributors_affiliations']
            cref = cmd['reference']
            if crrjsonfile not in citation_json:
                citation_json[crrjsonfile] = {cref: [caff]}
            else:
                pass

        with open(os.path.join(final_dir, "citation_list.json"), 'a') as f:
            json.dump(citation_json, f)

    @classmethod
    def get_contributors(cls, crr_dict):
        """
        Extract contributors name and affiliation from dictionary
        """

        # fill contributors field
        if "contributors_affiliations" in crr_dict:
            crr_contr = crr_dict["contributors_affiliations"]
            contributors = {
                'name': crr_contr, "message": "Data contributors: " +
                crr_contr}
        else:
            contributors = {}

        return contributors

    @classmethod
    def get_ljp(cls, crr_dict, volt_unit, ljpflag):
        """
        Extract the liquid junction potential (i.e. ljp)
        """

        ljp = {}

        # fill ljp (i.e. liquid junction potential) field
        if "liquid_junction_potential" in crr_dict and \
                "liquid_junction_potential_unit" in crr_dict:
            crr_ljp = crr_dict["liquid_junction_potential"]
            crr_ljp_u = crr_dict["liquid_junction_potential_unit"]

            if crr_ljp and crr_ljp[0] != 0 and crr_ljp_u:
                crr_ljp = crr_ljp * conversion_factor(volt_unit, crr_ljp_u)
                ljp = {
                    "ljp": {
                        "value": crr_ljp, "unit": crr_ljp_u,
                        "message": "Liquid junction potential value: " +
                        str(crr_ljp[0]), "computedflag": ljpflag}}

                return ljp

    @classmethod
    def get_holding_current(cls, crr_dict, amp_unit):
        """
        Fill holdcurr (i.e. holding current) field
        """
        hca = "holding_current"
        hcu = "holding_current_unit"

        holding_current = {}

        if hca and hcu in crr_dict:

            chca = crr_dict[hca]
            chcu = crr_dict[hcu]

            if chca and chcu:
                chca = chca[0] * manageConfig.conversion_factor(amp_unit, chcu)
                holding_current = {
                    "holdcurr": {
                        "value": [chca], "holdcurru": amp_unit,
                        "message": "Applied holding current: " + str(chca) +
                        " " + amp_unit
                    }
                }
        return holding_current


class manageConfig():
    """
    Manage configuration dictionary
    """

    # stimulus current units
    cu = ('a', 'da', 'ca', 'ma', 'ua', 'na', 'pa')

    # voltage unit
    vu = ('v', 'mv')

    # frequency unit
    fu = ('khz', 'hz')

    # all units
    all_units = cu + vu + fu

    @classmethod
    def conversion_factor(cls, unit_to, unit_from):
        """
        Extract conversion factors from unit_two to unit_one
        """

        conversion_table = {
            "v-mv": 1 / 1e3, "mv-v": 1e3, "a-da": 1 / 10, "a-ca": 1 / 1e2,
            "a-ma": 1 / 1e3, "a-ua": 1 / 1e6, "a-na": 1 / 1e9,
            "a-pa": 1 / 1e12, "da-a": 1 / 10, "da-ca": 10, "da-ma": 1e2,
            "da-ua": 1e5, "da-na": 1e8, "da-pa": 1e11, "ca-a": 1 / 1e2,
            "ca-da": 1 / 10, "ca-ma": 10, "ca-ua": 1e4, "ca-na": 1e7,
            "ca-pa": 1e10, "na-a": 1e9, "na-da": 1e8, "na-ca": 1e7,
            "na-ma": 1e6, "na-ua": 1e3, "na-pa": 1 / 1e3, "ma-a": 1e3,
            "ma-da": 1e2, "ma-ca": 10, "ma-ua": 1 / 1e3, "ma-na": 1e6,
            "ma-pa": 1e9, "ua-a": 1 / 1e6, "ua-da": 1 / 1e5, "ua-ca": 1 / 1e4,
            "ua-ma": 1 / 1e3, "ua-na": 1e3, "ua-pa": 1e6, "pa-a": 1e12,
            "pa-da": 1e11, "pa-ca": 1e10, "pa-ma": 1e9, "pa-ua": 1e6,
            "pa-na": 1e3, "khz-hz": 1 / 1e3, "hz-khz": 1e3,
        }

        conv_string = unit_to.lower() + "-" + unit_from.lower()

        if unit_to.lower() == unit_from.lower():
            return 1

        elif unit_to.lower() not in cls.all_units or \
                unit_from.lower() not in cls.all_units or \
                conv_string not in conversion_table:
            raise ValueError(
                "Given unit/s cannot be converted. Program is exiting")
        else:
            return conversion_table[conv_string]

    @classmethod
    def get_exclude_values(cls, cells_cellname, idx_file):
        """
        Extract stimulus values to be excluded and corresponding units from
        the configuration dictionary (i.e. 'config')
        """
        # extract stim traces to be excluded
        crr_exc = []
        crr_exc_u = []
        if "exclude" in cells_cellname and len(cells_cellname["exclude"]):

            exclude_values = cells_cellname["exclude"]

            if len(exclude_values) == 1:
                crr_exc = exclude_values[0]
                idx = 0
            elif len(exclude_values) - 1 >= idx_file:
                crr_exc = exclude_values[idx_file]
                idx = idx_file
            else:
                raise ValueError(
                    "'exclude' list length is shorter than " +
                    "the files list. Check your configuration. " +
                    "Program is exiting")

            if "exclude_unit" not in cells_cellname:
                crr_exc_u = 'nA'
            elif len(cells_cellname["exclude"]) == \
                    len(cells_cellname["exclude_unit"]):
                crr_exc_u = cells_cellname["exclude_unit"][idx]
            else:
                raise ValueError(
                    "'exclude' and 'exclude_unit' lists must " +
                    "have the same length. Program is exiting")

            conv_fact = manageConfig.conversion_factor('nA', crr_exc_u)
            crr_exc = [i * conv_fact for i in crr_exc]

        return [crr_exc, crr_exc_u]


class manageDicts():

    @classmethod
    def initialize_data_dict(cls):
        """
        Initialize the dictionary containing all data and info on the
        electrophysiological traces
        """

        data = OrderedDict()
        data['voltage'] = []
        data['current'] = []
        data['dt'] = []

        data['t'] = []
        data['ton'] = []
        data['toff'] = []
        data['tend'] = []
        data['amp'] = []
        data['hypamp'] = []
        data['filename'] = []

        return data

    @classmethod
    def fill_dict_single_trace(
            cls, data={}, voltage=0.0, current=0.0, dt=0.0, t=0.0, ton=0.0,
            toff=0.0, amp=0.0, hypamp=0.0, filename=""):

        data['voltage'].append(voltage)
        data['current'].append(current)
        data['dt'].append(dt)

        data['t'].append(t)
        data['tend'].append(t[-1])
        data['ton'].append(ton)
        data['toff'].append(toff)
        data['amp'].append(amp)
        data['hypamp'].append(hypamp)
        data['filename'].append(filename)

        return True
