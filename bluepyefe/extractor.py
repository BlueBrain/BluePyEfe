
import matplotlib
matplotlib.use('Agg', warn=True)
import matplotlib.pyplot as plt
import numpy
import sys
import efel
#import igorpy
import os
import fnmatch
import itertools
import collections
from itertools import cycle
import pprint
try:
    import cPickle as pickle
except:
    import pickle

import json
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(stream=sys.stdout)
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

import tools
import plottools
from extra import *
import sh

class Extractor(object):

    def __init__(self, mainname='PC', config=OrderedDict()):
        self.config = config

        self.path = config['path']
        self.cells = config['cells']
        self.features = config['features']

        for experiment in self.features:
            f = self.features[experiment]
            self.features[experiment] = sorted(set(f), key=lambda x: f.index(x))

        self.format = config['format']

        self.dataset = OrderedDict()
        self.dataset_mean = OrderedDict()

        try:
            sh.git('add', '-A')
            sh.git('commit', '-a', '-m \"Running feature extraction\"')
        except:
            pass

        self.githash = str(sh.git('rev-parse', '--short', 'HEAD')).rstrip()

        self.max_per_plot = 16

        self.options = config['options']
        if "relative" not in self.options:
            self.options["relative"] = False

        if "amp_min" not in self.options:
            self.options["amp_min"] = 0.001 # minimum current amplitude used

        if "peak_min" not in self.options:
            self.options["peak_min"] = 0.001 # minimum current amplitude used for spike detection

        if "target" not in self.options:
            self.options["target"] = [100., 150., 200., 250.]

        if "tolerance" not in self.options:
            self.options["tolerance"] = 10

        if "strict_stiminterval" not in self.options:
            self.options["strict_stiminterval"] = {'base': False}

        if isinstance(self.options["tolerance"], list) is False:
            self.options["tolerance"] =\
                        numpy.ones(len(self.options["target"]))\
                        * self.options["tolerance"]

        if "nanmean" not in self.options:
            self.options["nanmean"] = False

        if "nanmean_cell" not in self.options:
            self.options["nanmean_cell"] = True

        if "nangrace" not in self.options:
            self.options["nangrace"] = 0

        if "delay" not in self.options:
            self.options["delay"] = 0

        if "posttime" not in self.options:
            self.options["posttime"] = 200

        if "spike_threshold" not in self.options:
            self.options["spike_threshold"] = 2

        self.colors = OrderedDict()
        self.colors['b1'] = '#1F78B4' #377EB8
        self.colors['b2'] = '#A6CEE3'
        self.colors['g1'] = '#33A02C' #4DAF4A
        self.colors['g2'] = '#B2DF8A'
        self.colors['r1'] = '#E31A1C' #E41A1C
        self.colors['r2'] = '#FB9A99'
        self.colors['o1'] = '#FF7F00' #FF7F00
        self.colors['o2'] = '#FDBF6F'
        self.colors['p1'] = '#6A3D9A' #984EA3
        self.colors['p2'] = '#CAB2D6'

        self.colors['ye1'] = '#FFFF33'
        self.colors['br1'] = '#A65628'
        self.colors['pi1'] = '#F781BF'
        self.colors['gr1'] = '#999999'
        self.colors['k1'] = '#000000'

        self.markerlist = ['o', '*', '^', 'H', 'D', 's', 'p', '.', '8', '+']
        self.colorlist = [self.colors['b1'], self.colors['g1'], self.colors['r1'],
                            self.colors['o1'], self.colors['p1'], self.colors['ye1']]
        self.experiments = []

        self.mainname = mainname
        self.maindirname = mainname + os.sep # llb

        self.thresholds_per_cell = OrderedDict()
        self.hypamps_per_cell = OrderedDict()

        self.extra_features = ['spikerate_tau_jj', 'spikerate_drop',
                                'spikerate_tau_log', 'spikerate_tau_fit',
                                'spikerate_tau_slope']

    def newmeancell(self, a):
        if (self.options["nanmean_cell"] or
            (numpy.sum(numpy.isnan(a)) <= self.options["nangrace"])):
            return numpy.nanmean(a)
        else:
            return numpy.mean(a)

    def newstdcell(self, a):
        if (self.options["nanmean_cell"]
            or (numpy.sum(numpy.isnan(a)) <= self.options["nangrace"])):
            return numpy.nanstd(a)
        else:
            return numpy.std(a)


    def newmean(self, a):
        if (self.options["nanmean"] or
            (numpy.sum(numpy.isnan(a)) <= self.options["nangrace"])):
            return numpy.nanmean(a)
        else:
            return numpy.mean(a)

    def newstd(self, a):
        if (self.options["nanmean"]
            or (numpy.sum(numpy.isnan(a)) <= self.options["nangrace"])):
            return numpy.nanstd(a)
        else:
            return numpy.std(a)


    def create_dataset(self):

        logger.info(" Filling dataset")

        for i_cell, cellname in enumerate(self.cells):

            self.dataset[cellname] = OrderedDict()
            self.dataset[cellname]['v_corr'] = self.cells[cellname]['v_corr']

            if 'ljp' in self.cells[cellname]:
                ljp = self.cells[cellname]['ljp']
            else:
                ljp = 0
            self.dataset[cellname]['ljp'] = ljp

            dataset_cell_exp = OrderedDict()
            self.dataset[cellname]['experiments'] = dataset_cell_exp

            for i_exp, expname in enumerate(self.cells[cellname]['experiments']):

                files = self.cells[cellname]['experiments'][expname]['files']
                if len(files) > 0:
                    logger.debug(" Adding experiment %s", expname)

                    if expname not in self.experiments:
                        self.experiments.append(expname)


                    dataset_cell_exp[expname] = OrderedDict()

                    dataset_cell_exp[expname]['location'] =\
                        self.cells[cellname]['experiments'][expname]['location']

                    dataset_cell_exp[expname]['voltage'] = []
                    dataset_cell_exp[expname]['current'] = []
                    dataset_cell_exp[expname]['dt'] = []
                    dataset_cell_exp[expname]['filename'] = []

                    dataset_cell_exp[expname]['t'] = []
                    dataset_cell_exp[expname]['ton'] = []
                    dataset_cell_exp[expname]['tend'] = []
                    dataset_cell_exp[expname]['toff'] = []
                    dataset_cell_exp[expname]['amp'] = []
                    dataset_cell_exp[expname]['hypamp'] = []

                    for i_file, filename in enumerate(files):

                        data = self.process_file(filename, cellname, expname, i_file)
                        dataset_cell_exp[expname]['voltage'] += data['voltage']
                        dataset_cell_exp[expname]['current'] += data['current']
                        dataset_cell_exp[expname]['dt'] += data['dt']
                        dataset_cell_exp[expname]['filename'] += data['filename']

                        dataset_cell_exp[expname]['t'] += data['t']
                        dataset_cell_exp[expname]['ton'] += data['ton']
                        dataset_cell_exp[expname]['tend'] += data['tend']
                        dataset_cell_exp[expname]['toff'] += data['toff']
                        dataset_cell_exp[expname]['amp'] += data['amp']
                        dataset_cell_exp[expname]['hypamp'] += data['hypamp']


    def process_file(self, filename, cellname, expname, idx_file):

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

        ljp = self.dataset[cellname]['ljp']
        v_corr = self.dataset[cellname]['v_corr']

        # read stimulus features if present
        stim_feats = []
        if 'stim_feats' in self.cells[cellname]['experiments'][expname]:
            stim_feats = self.cells[cellname]['experiments'][expname]['stim_feats']

        if self.format == 'igor':

            import igorpy

            ordinal = filename['ordinal']
            logger.debug(" Adding igor file with ordinal %s", ordinal)

            i_file = self.path + filename['i_file']
            v_file = self.path + filename['v_file']

            notes, wave = igorpy.read(v_file)

            if 'v_unit' not in filename:
                v_unit = notes.xUnits
            else:
                v_unit = filename['v_unit']

            if v_unit == 'V':
                v = wave * 1e3  # mV
            elif v_unit == 'mV':
                v = wave  # mV
            else:
                raise Exception(
                    "Unit voltage not configured!")

            if 'dt' in filename:
                dt = filename['dt']
                if numpy.isclose(dt, notes.dx) is False:
                    raise Exception(
                        "Given stepsize %f does not match stepsize from wavenotes %f" % (dt, notes.dt))
            else:
                dt = notes.dx

            if 't_unit' not in filename:
                t_unit = notes.dUnits
            else:
                t_unit = filename['t_unit']

            if (t_unit == "") or (t_unit == "s"):
                dt = dt * 1e3 # convert to ms
            t = dt * numpy.arange(0, len(wave))

            notes, wave = igorpy.read(i_file)

            if 'i_unit' not in filename:
                i_unit = notes.xUnits
            else:
                i_unit = filename['i_unit']

            if i_unit == 'A':
                i = wave * 1e9 # nA
            elif i_unit == 'pA':
                i = wave * 1e-3 # nA
            else:
                raise Exception(
                    "Unit current not configured!")

            ton = self.options['onoff'][expname][0]
            toff = self.options['onoff'][expname][1]

            ion = int(ton/dt)

            if toff:
                ioff = int(toff/dt)
            else:
                ioff = False

            hypamp = numpy.mean( i[0:ion] )  # estimate hyperpolarization current
            iborder = int((ioff-ion)*0.1)  # 10% distance to measure step current
            # depolarization amplitude starting from hypamp!!

            if expname in ['APThreshold']:
                imax = numpy.argmax(i)
                toff = t[imax]
                trun = toff - ton
                ampoff = numpy.mean( i[int(imax-10./dt):imax] ) - hypamp
                amp = ampoff/trun * 2000. # extrapolate to get expected amplitude at 1 sec
                #amp = ampoff
            elif expname in ['H40S8']:
                amp = numpy.mean(i[i > 0.1])
            else:
                amp = numpy.mean( i[ion+iborder:ioff-iborder] ) - hypamp

            # clean voltage from transients
            if expname in ['IDRest', 'IDrest', 'IDthresh', 'IDdepol']:
                cut_start = int(ion+numpy.ceil(1.0/dt))
                v[ion:cut_start] = v[cut_start]
                cut_end0 = int(ioff-numpy.ceil(0.5/dt))
                cut_end1 = int(ioff+numpy.ceil(2.0/dt))
                v[cut_end0:cut_end1] = v[cut_end1]

            # delete second pulse
            elif expname in ['SpikeRec']:
                t = t[:int(50./dt)]
                v = v[:int(50./dt)]
                i = i[:int(50./dt)]

            # normalize membrane potential to known value
            if v_corr:
                v = v - numpy.mean(v[0:ion]) + v_corr

            v = v - ljp # correct junction potential

            data['voltage'].append(v)
            data['current'].append(i)
            data['dt'].append(dt)

            data['t'].append(t)
            data['tend'].append(t[-1])
            data['ton'].append(ton)
            data['toff'].append(toff)
            data['amp'].append(amp)
            data['hypamp'].append(hypamp)
            data['filename'].append(ordinal)

        elif self.format == 'axon':

            logger.debug(" Adding axon file %s", filename)

            from neo import io
            f = self.path + cellname + os.sep + filename + '.abf'
            r = io.AxonIO(filename = f) #
            header = r.read_header() # read file header
            sampling_rate = 1.e6 / header['protocol']['fADCSequenceInterval'] # read sampling rate

            dt = 1./int(sampling_rate) * 1e3
            version = header['fFileVersionNumber'] # read file version
            bl = r.read_block(lazy=False, cascade=True)
            all_stims = []
            if stim_feats:
                res = self.stim_feats_from_meta(stim_feats, len(bl.segments), idx_file)
                if res[0]:
                    all_stims = res[1]
                else:
                    print(res[1])
            if not all_stims:
                res = self.stim_feats_from_header(header)
                if res[0]:
                    all_stims = res[1]
                else:
                    pprint.pprint("No valid stimulus was found in metadata or files. Skipping current file")
                    return

            # for all segments in file
            for i_seg, seg in enumerate(bl.segments):

                voltage = numpy.array(seg.analogsignals[0]).astype(numpy.float64)
                #current = numpy.array(seg.analogsignals[1]).astype(numpy.float64)
                #dt = 1./int(seg.analogsignals[0].sampling_rate) * 1e3

                t = numpy.arange(len(voltage)) * dt

                # when does voltage change
                #c_changes = numpy.where( abs(numpy.gradient(current, 1.)) > 0.0 )[0]

                # detect on and off of current
                #c_changes2 = numpy.where( abs(numpy.gradient(c_changes, 1.)) > 10.0 )[0]

                #ion = c_changes[c_changes2[0]]
                #ioff = c_changes[-1]
                #ton = ion * dt
                #toff = ioff * dt

                ton = all_stims[i_seg][1]
                toff = all_stims[i_seg][2]
                ion = int(ton / dt)
                ioff = int(toff / dt)
                amp = numpy.float64(all_stims[i_seg][3])

                current = []
                current = numpy.zeros(len(voltage))
                current[ion:ioff] = amp

                # estimate hyperpolarization current
                hypamp = numpy.mean( current[0:ion] )

                # 10% distance to measure step current
                iborder = int((ioff-ion)*0.1)

                # depolarization amplitude
                #amp = numpy.mean( current[ion+iborder:ioff-iborder] )
                voltage_dirty = voltage[:]

                # clean voltage from transients
                voltage[ion:ion+int(numpy.ceil(0.4/dt))] = voltage[ion+int(numpy.ceil(0.4/dt))]
                voltage[ioff:ioff+int(numpy.ceil(0.4/dt))] = voltage[ioff+int(numpy.ceil(0.4/dt))]

                # normalize membrane potential to known value (given in UCL excel sheet)
                if v_corr:
                    if len(v_corr) == 1 and v_corr[0] != 0.0:
                        voltage = voltage - numpy.mean(voltage[0:ion]) + v_corr[0]
                    elif len(v_corr) - 1 >= idx_file and v_corr[idx_file] != 0.0:
                        voltage = voltage - numpy.mean(voltage[0:ion]) + v_corr[idx_file]

                voltage = voltage - ljp

                # clip spikes after stimulus so they are not analysed
                voltage[ioff:] = numpy.clip(voltage[ioff:], -300, -40)

                if ('exclude' in self.cells[cellname] and
                    any(abs(self.cells[cellname]['exclude'][idx_file] - amp) < 1e-4)):
                    continue # llb

                else:
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


        elif self.format == 'csv_lccr':

            if isinstance(filename, str) is False:
                raise Exception('Please provide a string with filename of csv file')

            filename = self.path + '/' + filename + '.txt'

            exp_options = self.cells[cellname]['experiments'][expname]

            if (('dt' not in exp_options) or
                ('amplitudes' not in exp_options) or
                ('hypamp' not in exp_options) or
                ('ton' not in exp_options) or
                ('toff' not in exp_options)):
                raise Exception('Please provide additional options for LCCR csv')

            dt = exp_options['dt']
            hypamp = exp_options['hypamp']
            ton = exp_options['startstop'][0]
            toff = exp_options['startstop'][1]
            amplitudes = exp_options['amplitudes']

            import csv
            with open(filename, 'rb') as f:
                reader = csv.reader(f, delimiter='\t')
                columns = zip(*reader)
                length = numpy.shape(columns)[1]

                for ic, column in enumerate(columns):

                    voltage = numpy.zeros(length)
                    for istr, string in enumerate(column):
                        if (string != "-") and (string != ""):
                            voltage[istr] = float(string)

                    t = numpy.arange(len(voltage)) * dt
                    amp = amplitudes[ic]

                    voltage = voltage - ljp # correct liquid junction potential

                    # remove last 100 ms
                    voltage = voltage[0:int(-100./dt)]
                    t = t[0:int(-100./dt)]
                    current = None

                    if ('exclude' in self.cells[cellname]
                        and any(abs(self.cells[cellname]['exclude'][i_file] - amp) < 1e-4)):
                        continue #llb

                    else:
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

        return data

    # author Luca Leonardo Bologna
    def stim_feats_from_meta(self, stim_feats, num_segments, idx_file):
        if not stim_feats:
            return (0, "Empty metadata in file")
        elif len(stim_feats) - 1 < idx_file and len(stim_feats) !=1:
            return (0, "Stimulus dictionaries are different from the number of files")
        else:
            # array for storing all stimulus features
            all_stim_feats = []

            # for every segment in the axon file
            for i in range(num_segments):

                # read current stimulus dict
                if len(stim_feats) == 1:
                    crr_dict = stim_feats[0]
                else:
                    crr_dict = stim_feats[idx_file]

                # read stimulus information
                ty = str(crr_dict['stimulus_type'])
                tu = crr_dict['stimulus_time_unit']
                st = crr_dict['stimulus_start']
                en = crr_dict['stimulus_end']
                u = str(crr_dict['stimulus_unit'])
                fa = float(format(crr_dict['stimulus_first_amplitude'], '.3f'))
                inc = float(format(crr_dict['stimulus_increment'], '.3f'))
                ru = crr_dict['sampling_rate_unit']
                r = crr_dict['sampling_rate']
                if tu == 's':
                    st = st * 1e3
                    en = en * 1e3
                # compute current stimulus amplitude
                crr_val = float(format(fa + inc * float(format(i, '.3f')), '.3f'))
                crr_stim_feats = (ty, st, en, crr_val, u)

                # store current tuple
                all_stim_feats.append(crr_stim_feats)
            return (1, all_stim_feats)


    # author Luca Leonardo Bologna
    def stim_feats_from_header(self, header):
        sampling_rate = 1.e6 / header['protocol']['fADCSequenceInterval'] # read sampling rate
        version = header['fFileVersionNumber'] # read file version

        # extract protocol for version >=.2
        if version >= 2.:
            #prot = r.read_protocol() # read protocol
            dictEpochInfoPerDAC = header['dictEpochInfoPerDAC'] # read info for DAC

            # if field is empty
            if not (dictEpochInfoPerDAC):
                return (0, "No 'dictEpochInfoPerDAC' field")

            # if field is not empty, read all stimulus segments
            else:
                valid_epoch_dicts = [k for k, v in dictEpochInfoPerDAC.iteritems() if bool(v)]

                # if more than one channel is activated for the stimulus
                # or a number of epochs different than 3 is found
                if len(valid_epoch_dicts) != 1 or len(dictEpochInfoPerDAC[0]) != 3:
                    return (0, 'Exiting. More than one channel used for stimulation')
                else:
                    stim_epochs = dictEpochInfoPerDAC[k] # read all stimulus epochs
                    stim_ch_info = [(i['DACChNames'], i['DACChUnits'], i['nDACNum']) for i in header['listDACInfo'] if bool(i['nWaveformEnable'])] # read enabled waveforms

                    # if epoch initial levels and increment are not compatible with a step stimulus
                    if (stim_epochs[0]['fEpochInitLevel'] != stim_epochs[2]['fEpochInitLevel'] or
                        stim_epochs[0]['fEpochLevelInc'] != stim_epochs[2]['fEpochLevelInc'] or
                        float(format(stim_epochs[0]['fEpochLevelInc'], '.3f')) != 0 or
                        (len(stim_ch_info) != 1 or stim_ch_info[0][2] != k)):
                            # return 0 with message
                            return (0, "A stimulus different from the steps has been detected")
                    else:
                        ty = "step"
                        u = stim_ch_info[0][1]
                        nADC = header['sections']['ADCSection']['llNumEntries'] # number of ADC channels
                        nDAC = header['sections']['DACSection']['llNumEntries'] # number of DAC channels
                        nSam = header['protocol']['lNumSamplesPerEpisode']/nADC # number of samples per episode
                        nEpi = header['lActualEpisodes'] # number of actual episodes

                        e_zero = header['dictEpochInfoPerDAC'][stim_ch_info[0][2]][0] # read first stimulus epoch
                        e_one = header['dictEpochInfoPerDAC'][stim_ch_info[0][2]][1] # read second stimulus epoch
                        e_two = header['dictEpochInfoPerDAC'][stim_ch_info[0][2]][2] # read third stimulus epoch

                        i_last = int(nSam*15625/10**6) # index of stimulus beginning

                        all_stim_info = [] # create array for all stimulus info

                        e_one_inc = float(format(e_one['fEpochLevelInc'] , '.3f')) # step increment
                        e_one_init_level = float(format(e_one['fEpochInitLevel'] , '.3f')) # step initial level

                        # for every episode, compute stimulus start, stimulus end, stimulus value
                        for epiNum in range(nEpi):
                            st = i_last + e_zero['lEpochInitDuration'] + e_zero['lEpochDurationInc'] * epiNum
                            en = st + e_one['lEpochInitDuration'] +  e_one['lEpochDurationInc'] * epiNum
                            crr_val_full = float(format(e_one_init_level + e_one_inc * epiNum, '.3f'))
                            crr_val = float(format(crr_val_full, '.3f'))
                            st = 1/sampling_rate * st * 1e3
                            en = 1/sampling_rate * en * 1e3
                            all_stim_info.append((ty, st, en, crr_val, u))
                        return (1, all_stim_info)


    def plt_traces(self):

        logger.info(" Plotting traces")
        tools.makedir(self.maindirname)

        for i_cell, cellname in enumerate(self.dataset):

            dirname = self.maindirname+cellname
            tools.makedir(dirname)
            dataset_cell_exp = self.dataset[cellname]['experiments']

            for i_exp, expname in enumerate(dataset_cell_exp):

                voltages = dataset_cell_exp[expname]['voltage']
                currents = dataset_cell_exp[expname]['current']
                amps = dataset_cell_exp[expname]['amp']
                ts = dataset_cell_exp[expname]['t']
                filenames = dataset_cell_exp[expname]['filename']

                # old:
                # voltages = list(itertools.chain.from_iterable(voltages))
                colors = []

                markercycler = cycle(self.markerlist)
                colorcycler = cycle(self.colorlist)

                color_dict = {u:next(colorcycler) for u in list(set(filenames))}
                colors = [color_dict[f] for f in filenames]

                isort = numpy.argsort(amps)
                amps = numpy.array(amps)[isort]
                ts = numpy.array(ts)[isort]
                voltages = numpy.array(voltages)[isort]
                currents = numpy.array(currents)[isort]
                colors = numpy.array(colors)[isort]
                filenames = numpy.array(filenames)[isort]

                n_plot = len(voltages)

                if n_plot <= self.max_per_plot:
                    frames = n_plot
                    n_fig = 1
                else:
                    frames = self.max_per_plot
                    n_fig = int(numpy.ceil(n_plot/float(self.max_per_plot)))

                axs = []
                figs = OrderedDict()

                axs_c = []
                figs_c = OrderedDict()

                for i_fig in range(n_fig):
                    figname = cellname.split('/')[-1] + "_" + expname + "_" + str(i_fig)
                    axs = plottools.tiled_figure(figname, frames=frames, columns=2,
                                    figs=figs, axs=axs,
                                    top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.2)

                    figname_c = cellname.split('/')[-1] + "_" + expname + "_" + str(i_fig) + "_i"
                    axs_c = plottools.tiled_figure(figname_c, frames=frames, columns=2,
                                    figs=figs_c, axs=axs_c,
                                    top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.2)

                for i_plot in range(n_plot):
                    axs[i_plot].plot(ts[i_plot], voltages[i_plot], color=colors[i_plot], clip_on=False)
                    axs[i_plot].set_title(cellname + " " + expname + " amp:" + str(amps[i_plot]) + " file:" + filenames[i_plot])

                    axs_c[i_plot].plot(ts[i_plot], currents[i_plot], color=colors[i_plot], clip_on=False)
                    axs_c[i_plot].set_title(cellname + " " + expname + " amp:" + str(amps[i_plot]) + " file:" + filenames[i_plot])

                #plt.show()

                for i_fig, figname in enumerate(figs):
                    fig = figs[figname]
                    fig['fig'].savefig(dirname + '/' + figname + '.pdf', dpi=300)
                    plt.close(fig['fig'])

                for i_fig, figname in enumerate(figs_c):
                    fig = figs_c[figname]
                    fig['fig'].savefig(dirname + '/' + figname + '.pdf', dpi=300)
                    plt.close(fig['fig'])


    def extract_features(self, threshold=-20):

        logger.info(" Extracting features")

        #efel.Settings.derivative_threshold = 5.0
        efel.setThreshold(threshold)
        logger.info(" Setting threshold to %f", threshold)


        for i_cell, cellname in enumerate(self.dataset):

            dataset_cell_exp = self.dataset[cellname]['experiments']
            for i_exp, expname in enumerate(dataset_cell_exp):

                if expname in self.options["strict_stiminterval"].keys():
                    strict_stiminterval = self.options["strict_stiminterval"][expname]
                else:
                    strict_stiminterval = self.options["strict_stiminterval"]['base']
                efel.setIntSetting("strict_stiminterval",strict_stiminterval)

                ts = dataset_cell_exp[expname]['t']
                voltages = dataset_cell_exp[expname]['voltage']
                #currents = dataset_cell_exp[expname]['current']
                #tends = dataset_cell_exp[expname]['tend']
                tons = dataset_cell_exp[expname]['ton']
                toffs = dataset_cell_exp[expname]['toff']
                amps = dataset_cell_exp[expname]['amp']
                #filenames = dataset_cell_exp[expname]['filename']

                features_all = self.features[expname] + ['peak_time']

                if 'threshold' in self.cells[cellname]['experiments'][expname]:
                    threshold = self.cells[cellname]['experiments'][expname]['threshold']
                    efel.setThreshold(threshold)

                dataset_cell_exp[expname]['features'] = OrderedDict()
                for feature in features_all:
                    dataset_cell_exp[expname]['features'][feature] = []
                dataset_cell_exp[expname]['features']['numspikes'] = []

                # iterate over all voltages individually to get features
                # and correct them if needed
                for i_seg in range(len(voltages)):

                    trace = OrderedDict()
                    trace['T'] = ts[i_seg]
                    trace['V'] = voltages[i_seg]
                    trace['stim_start'] = [tons[i_seg]]
                    trace['stim_end'] = [toffs[i_seg]]
                    traces = [trace]
                    amp = amps[i_seg]

                    efel.setDoubleSetting('stimulus_current', amp)

                    features_all_ = [f for f in features_all if f not in self.extra_features]

                    fel_vals = efel.getFeatureValues(traces, features_all_, raise_warnings=False)

                    peak_times = fel_vals[0]['peak_time']

                    for feature in features_all:

                        if feature == 'peak_time':
                            if (len(peak_times) > 0) and amp >= self.options["peak_min"]:
                                numspike = len(peak_times)
                            else:
                                numspike = 0

                        elif feature == 'spikerate_tau_jj':
                            if len(peak_times) > 4:
                                f = spikerate_tau_jj(peak_times)
                            else:
                                f = None

                        elif feature == 'spikerate_drop':
                            if len(peak_times) > 4:
                                f = spikerate_drop(peak_times)
                            else:
                                f = None

                        elif feature == 'spikerate_tau_log':
                            if len(peak_times) > 4:
                                f = spikerate_tau_log(peak_times)
                            else:
                                f = None

                        elif feature == 'spikerate_tau_fit':
                            if len(peak_times) > 4:
                                f = spikerate_tau_fit(peak_times)
                            else:
                                f = None

                        elif feature == 'spikerate_tau_slope':
                            if len(peak_times) > 4:
                                f = spikerate_tau_slope(peak_times)
                            else:
                                f = None

                        else:
                            f = fel_vals[0][feature]

                        if abs(amp) < self.options["amp_min"]:
                            f = float('nan')
                        elif f is not None:
                            f = numpy.mean(f)
                        else:
                            f = float('nan')

                        if "trace_check" in self.options and self.options["trace_check"] is False:
                            pass
                        else:
                            # exclude any activity outside stimulus (20 ms grace period)
                            if (any(numpy.atleast_1d(peak_times) < trace['stim_start'][0]) or
                               any(numpy.atleast_1d(peak_times) > trace['stim_end'][0]+20)):
                               f = float('nan')

                        dataset_cell_exp[expname]['features'][feature].append(f)

                    dataset_cell_exp[expname]['features']['numspikes'].append(numspike)


    def mean_features(self):

        # mean for each cell
        for i_cell, cellname in enumerate(self.dataset):

            dataset_cell_exp = self.dataset[cellname]['experiments']

            # compute threshold based on some experiments
            if 'expthreshold' in self.options:
                hypamp = []
                amp = []
                numspikes = []

                for i_exp, expname in enumerate(dataset_cell_exp):
                    if expname in self.options['expthreshold']: # use to determine threshold
                        hypamp = hypamp + dataset_cell_exp[expname]['hypamp']
                        amp = amp + dataset_cell_exp[expname]['amp']
                        numspikes = numspikes + dataset_cell_exp[expname]['features']['numspikes']

                mean_hypamp = self.newmeancell(numpy.array(hypamp))
                amp_threshold = self.get_threshold(amp, numspikes)

                self.thresholds_per_cell[cellname] = amp_threshold
                self.hypamps_per_cell[cellname] = mean_hypamp

                logger.info(" %s threshold amplitude: %f hypamp: %f", cellname, amp_threshold, mean_hypamp)

            else:
                self.thresholds_per_cell[cellname] = None
                self.hypamps_per_cell[cellname] = None

            for i_exp, expname in enumerate(dataset_cell_exp):

                # define
                dataset_cell_exp[expname]['mean_amp'] = OrderedDict()
                dataset_cell_exp[expname]['mean_amp_rel'] = OrderedDict()
                dataset_cell_exp[expname]['std_amp'] = OrderedDict()
                dataset_cell_exp[expname]['std_amp_rel'] = OrderedDict()

                dataset_cell_exp[expname]['mean_hypamp'] = OrderedDict()
                dataset_cell_exp[expname]['std_hypamp'] = OrderedDict()

                dataset_cell_exp[expname]['mean_features'] = OrderedDict()
                dataset_cell_exp[expname]['std_features'] = OrderedDict()

                for feature in self.features[expname]:
                    dataset_cell_exp[expname]['mean_features'][feature] = OrderedDict()
                    dataset_cell_exp[expname]['std_features'][feature] = OrderedDict()

                ton = dataset_cell_exp[expname]['ton']
                toff = dataset_cell_exp[expname]['toff']
                tend = dataset_cell_exp[expname]['tend']

                ton = self.newmeancell(numpy.array(ton))
                toff = self.newmeancell(numpy.array(toff))
                tend = self.newmeancell(numpy.array(tend))

                hypamp = dataset_cell_exp[expname]['hypamp']
                amp = dataset_cell_exp[expname]['amp']
                numspikes = dataset_cell_exp[expname]['features']['numspikes']
                feature_array = dataset_cell_exp[expname]['features']

                if self.options["relative"]:
                    amp_threshold = self.thresholds_per_cell[cellname]
                    amp_rel = numpy.array(amp/amp_threshold * 100.)
                else:
                    amp_rel = numpy.array(amp)

                # absolute amplitude not relative to hypamp
                amp_abs = numpy.abs(numpy.array(dataset_cell_exp[expname]['amp']) +
                            numpy.array(dataset_cell_exp[expname]['hypamp']))

                i_noinput = numpy.argmin(amp_abs)

                # save amplitude results
                for ti, target in enumerate(self.options["target"]):

                    if target == 'noinput':
                        idx = numpy.array(i_noinput)
                    elif target == 'all':
                        idx = numpy.ones(len(amp), dtype=bool)
                    else:
                        idx = numpy.array((amp_rel >= (target - self.options["tolerance"][ti])) &\
                                (amp_rel <= (target + self.options["tolerance"][ti])))

                    amp_target = numpy.atleast_1d(numpy.array(amp)[idx])
                    # equal to amp_target if amplitude not measured relative to threshold
                    amp_rel_target = numpy.atleast_1d(numpy.array(amp_rel)[idx])
                    hypamp_target = numpy.atleast_1d(numpy.array(hypamp)[idx])

                    if len(amp_target) > 0:

                        if (target == 'noinput'):
                            if (amp_abs[i_noinput] < 0.01):
                                meanamp_target = self.newmeancell(amp_target)
                                stdamp_target = self.newstdcell(amp_target)
                                meanamp_rel_target = self.newmeancell(amp_rel_target)
                                stdamp_rel_target = self.newstdcell(amp_rel_target)
                                meanhypamp_target = self.newmeancell(hypamp_target)
                                stdhypamp_target = self.newstdcell(hypamp_target)
                            else:
                                continue

                        else:
                            meanamp_target = self.newmeancell(amp_target)
                            stdamp_target = self.newstdcell(amp_target)
                            meanamp_rel_target = self.newmeancell(amp_rel_target)
                            stdamp_rel_target = self.newstdcell(amp_rel_target)
                            meanhypamp_target = self.newmeancell(hypamp_target)
                            stdhypamp_target = self.newstdcell(hypamp_target)

                        dataset_cell_exp[expname]['mean_amp'][str(target)] = meanamp_target
                        dataset_cell_exp[expname]['std_amp'][str(target)] = stdamp_target

                        dataset_cell_exp[expname]['mean_amp_rel'][str(target)] = meanamp_rel_target
                        dataset_cell_exp[expname]['std_amp_rel'][str(target)] = stdamp_rel_target

                        dataset_cell_exp[expname]['mean_hypamp'][str(target)] = meanhypamp_target
                        dataset_cell_exp[expname]['std_hypamp'][str(target)] = stdhypamp_target

                dataset_cell_exp[expname]['mean_ton'] = ton
                dataset_cell_exp[expname]['mean_toff'] = toff
                dataset_cell_exp[expname]['mean_tend'] = tend

                for fi, feature in enumerate(self.features[expname]):
                    feat_vals = numpy.array(feature_array[feature])
                    for ti, target in enumerate(self.options["target"]):

                        if target == 'noinput':
                            idx = numpy.array(i_noinput)
                        elif target == 'all':
                            idx = numpy.ones(len(amp), dtype=bool)
                        else:
                            idx = numpy.array((amp_rel >= (target - self.options["tolerance"][ti])) &\
                                    (amp_rel <= (target + self.options["tolerance"][ti])))

                        feat = numpy.atleast_1d(numpy.array(feat_vals)[idx])

                        #
                        if len(feat) > 0:

                            if (target == 'noinput'):
                                if (amp_abs[i_noinput] < 0.01):
                                    meanfeat = self.newmeancell(feat)
                                    stdfeat = self.newstdcell(feat)
                                else:
                                    continue
                            else:
                                meanfeat = self.newmeancell(feat)
                                stdfeat = self.newstdcell(feat)

                            dataset_cell_exp[expname]['mean_features'][feature][str(target)] = meanfeat
                            dataset_cell_exp[expname]['std_features'][feature][str(target)] = stdfeat


        # mean for all cells
        for i_exp, expname in enumerate(self.experiments):

            #collect everything in global structure
            self.dataset_mean[expname] = OrderedDict()
            self.dataset_mean[expname]['amp'] = OrderedDict()
            self.dataset_mean[expname]['amp_rel'] = OrderedDict()
            self.dataset_mean[expname]['hypamp'] = OrderedDict()
            self.dataset_mean[expname]['ton'] = []
            self.dataset_mean[expname]['toff'] = []
            self.dataset_mean[expname]['tend'] = []
            self.dataset_mean[expname]['features'] = OrderedDict()
            self.dataset_mean[expname]['cell_std_features'] = OrderedDict()
            self.dataset_mean[expname]['ncells'] = OrderedDict()

            for feature in self.features[expname]:
                self.dataset_mean[expname]['features'][feature] = OrderedDict()
                self.dataset_mean[expname]['cell_std_features'][feature] = OrderedDict()
                self.dataset_mean[expname]['ncells'][feature] = OrderedDict()
                for target in self.options["target"]:
                    self.dataset_mean[expname]['features'][feature][str(target)] = []
                    self.dataset_mean[expname]['cell_std_features'][feature][str(target)] = []

            for target in self.options["target"]:
                self.dataset_mean[expname]['amp'][str(target)] = []
                self.dataset_mean[expname]['amp_rel'][str(target)] = []
                self.dataset_mean[expname]['hypamp'][str(target)] = []

            for i_cell, cellname in enumerate(self.dataset):

                #tools.print_dict(self.dataset[cellname])
                dataset_cell_exp = self.dataset[cellname]['experiments']

                if expname in dataset_cell_exp:
                    self.dataset_mean[expname]['location'] =\
                            dataset_cell_exp[expname]['location']

                    ton = dataset_cell_exp[expname]['mean_ton']
                    self.dataset_mean[expname]['ton'].append(ton)

                    toff = dataset_cell_exp[expname]['mean_toff']
                    self.dataset_mean[expname]['toff'].append(toff)

                    tend = dataset_cell_exp[expname]['mean_tend']
                    self.dataset_mean[expname]['tend'].append(tend)

                    for target in self.options["target"]:
                        if str(target) in dataset_cell_exp[expname]['mean_amp']:
                            amp = dataset_cell_exp[expname]['mean_amp'][str(target)]
                            self.dataset_mean[expname]['amp'][str(target)].append(amp)
                            amp_rel = dataset_cell_exp[expname]['mean_amp_rel'][str(target)]
                            self.dataset_mean[expname]['amp_rel'][str(target)].append(amp_rel)
                            hypamp = dataset_cell_exp[expname]['mean_hypamp'][str(target)]
                            self.dataset_mean[expname]['hypamp'][str(target)].append(hypamp)

                    for feature in self.features[expname]:
                        for target in self.options["target"]:
                            if str(target) in dataset_cell_exp[expname]['mean_features'][feature]:
                                result = dataset_cell_exp[expname]['mean_features'][feature][str(target)]
                                self.dataset_mean[expname]['features'][feature][str(target)].append(result)
                                cell_std_result = dataset_cell_exp[expname]['std_features'][feature][str(target)]
                                self.dataset_mean[expname]['cell_std_features'][feature][str(target)].append(cell_std_result)

            #create means
            self.dataset_mean[expname]['mean_amp'] = OrderedDict()
            self.dataset_mean[expname]['mean_amp_rel'] = OrderedDict()
            self.dataset_mean[expname]['mean_hypamp'] = OrderedDict()
            self.dataset_mean[expname]['std_amp'] = OrderedDict()
            self.dataset_mean[expname]['std_amp_rel'] = OrderedDict()
            self.dataset_mean[expname]['std_hypamp'] = OrderedDict()
            self.dataset_mean[expname]['mean_features'] = OrderedDict()
            self.dataset_mean[expname]['std_features'] = OrderedDict()

            for feature in self.features[expname]:
                self.dataset_mean[expname]['mean_features'][feature] = OrderedDict()
                self.dataset_mean[expname]['std_features'][feature] = OrderedDict()

            ton = self.dataset_mean[expname]['ton']
            toff = self.dataset_mean[expname]['toff']
            tend = self.dataset_mean[expname]['tend']

            self.dataset_mean[expname]['mean_ton'] = self.newmean(ton)
            self.dataset_mean[expname]['mean_toff'] = self.newmean(toff)
            self.dataset_mean[expname]['mean_tend'] = self.newmean(tend)

            for target in self.options["target"]:
                amp = self.dataset_mean[expname]['amp'][str(target)]
                amp_rel = self.dataset_mean[expname]['amp_rel'][str(target)]
                hypamp = self.dataset_mean[expname]['hypamp'][str(target)]

                self.dataset_mean[expname]['mean_amp'][str(target)] = self.newmean(amp)
                self.dataset_mean[expname]['mean_amp_rel'][str(target)] = self.newmean(amp_rel)
                self.dataset_mean[expname]['mean_hypamp'][str(target)] = self.newmean(hypamp)
                self.dataset_mean[expname]['std_amp'][str(target)] = self.newstd(amp)
                self.dataset_mean[expname]['std_amp_rel'][str(target)] = self.newstd(amp_rel)
                self.dataset_mean[expname]['std_hypamp'][str(target)] = self.newstd(hypamp)

            for feature in self.features[expname]:
                for target in self.options["target"]:
                    feat = self.dataset_mean[expname]['features'][feature][str(target)]
                    cell_std_feat = self.dataset_mean[expname]['cell_std_features'][feature][str(target)]
                    # added by Luca Leonardo Bologna to handle the case in which only one feature value is present at this point
                    if len(feat) == 1:
                        if cell_std_feat != 0.0:
                            self.dataset_mean[expname]['mean_features'][feature][str(target)] = feat[0]
                            self.dataset_mean[expname]['std_features'][feature][str(target)] = cell_std_feat[0]
                    else:
                       self.dataset_mean[expname]['mean_features'][feature][str(target)] = self.newmean(feat)
                       self.dataset_mean[expname]['std_features'][feature][str(target)] = self.newstd(feat)
                    self.dataset_mean[expname]['ncells'][feature][str(target)] = numpy.sum(numpy.invert(numpy.isnan(numpy.atleast_1d(feat))))


    def get_threshold(self, amp, numspikes):

        isort = numpy.argsort(amp)
        amps_sort = numpy.array(amp)[isort]
        numspikes_sort = numpy.array(numspikes)[isort]
        i_threshold = numpy.where(numspikes_sort>=self.options["spike_threshold"])[0][0]
        amp_threshold = amps_sort[i_threshold]

        return amp_threshold


    def plt_features(self):

        logger.info(" Plotting features")
        figs = OrderedDict()

        markercyclercell = cycle(self.markerlist)
        colorcyclercell = cycle(self.colorlist)

        for i_cell, cellname in enumerate(self.dataset):

            cellfigs = OrderedDict()

            dirname = self.maindirname+cellname
            tools.makedir(dirname)

            colorcell = next(colorcyclercell)
            markercell = next(markercyclercell)
            dataset_cell_exp = self.dataset[cellname]['experiments']

            for i_exp, expname in enumerate(dataset_cell_exp):

                figname = "features_" + expname
                if figname not in figs:
                    plottools.tiled_figure(figname, frames=len(self.features[expname]),
                                    columns=3, figs=figs, dirname=self.maindirname,
                                    top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.3)


                figname = "features_" + cellname.split('/')[-1] + "_" + expname
                axs_cell = plottools.tiled_figure(figname, frames=len(self.features[expname]),
                                    columns=3, figs=cellfigs, dirname=dirname,
                                    top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.3)

                amp = dataset_cell_exp[expname]['amp']
                numspikes = dataset_cell_exp[expname]['features']['numspikes']
                feature_array = dataset_cell_exp[expname]['features']
                filenames = dataset_cell_exp[expname]['filename']

                markercycler = cycle(self.markerlist)
                colorcycler = cycle(self.colorlist)

                colormarker_dict = {u:[next(colorcycler),next(markercycler)] for u in list(set(filenames))}

                if self.options["relative"]:
                    amp_threshold = self.thresholds_per_cell[cellname]
                    amp_rel = amp/amp_threshold * 100.
                else:
                    amp_rel = amp

                for fi, feature in enumerate(self.features[expname]):

                    feat_vals = numpy.array(feature_array[feature])

                    for i_feat, feat_vals_ in enumerate(feat_vals):
                        color = colormarker_dict[filenames[i_feat]][0]
                        marker = colormarker_dict[filenames[i_feat]][1]

                        amp_rel_ = amp_rel[i_feat]
                        is_not_nan = ~numpy.isnan(feat_vals_)

                        axs_cell[fi].plot(amp_rel_[is_not_nan], feat_vals_[is_not_nan], "",
                                    linestyle='None',
                                    marker=marker, color=color, markersize=5, zorder=1,
                                    linewidth=1, markeredgecolor = 'none', clip_on=False)
                        #axs_cell[fi].set_xticks(self.options["target"])
                        #axs_cell[fi].set_xticklabels(())
                        axs_cell[fi].set_title(feature)

                        figname = "features_" + expname
                        figs[figname]['axs'][fi].plot(amp_rel_[is_not_nan], feat_vals_[is_not_nan], "",
                                    linestyle='None',
                                    marker=markercell, color=colorcell, markersize=3, zorder=1,
                                    linewidth=1, markeredgecolor = 'none', clip_on=False)
                        #figs[figname]['axs'][fi].set_xticks(self.options["target"])
                        figs[figname]['axs'][fi].set_title(feature)

                    if 'mean_features' in dataset_cell_exp[expname]:

                        amp_rel_list = []
                        mean_list = []
                        std_list = []
                        for target in self.options["target"]:
                            if str(target) in dataset_cell_exp[expname]['mean_amp_rel']:
                                amp_rel_list.append(dataset_cell_exp[expname]['mean_amp_rel'][str(target)])
                                mean_list.append(dataset_cell_exp[expname]['mean_features'][feature][str(target)])
                                std_list.append(dataset_cell_exp[expname]['std_features'][feature][str(target)])

                        mean_array = numpy.array(mean_list)
                        std_array = numpy.array(std_list)
                        amp_rel_array = numpy.array(amp_rel_list)

                        e = axs_cell[fi].errorbar(amp_rel_array, mean_array, yerr=std_array, marker='s', color='k',
                                    linewidth=1, markersize=6, zorder=10, clip_on = False)
                        #axs_cell[fi].set_xticks(self.options["target"])
                        for b in e[1]:
                            b.set_clip_on(False)

            # close singe cell figures
            for i_fig, figname in enumerate(cellfigs):
                fig = cellfigs[figname]
                fig['fig'].savefig(fig['dirname'] + '/' + figname + '.pdf', dpi=300)
                plt.close(fig['fig'])


        for i_exp, expname in enumerate(self.experiments):

            if expname in self.dataset_mean:
                for fi, feature in enumerate(self.features[expname]):
                    amp_rel_list = []
                    mean_list = []
                    std_list = []
                    for target in self.options["target"]:
                        if str(target) in self.dataset_mean[expname]['mean_amp_rel']:
                            amp_rel_list.append(self.dataset_mean[expname]['mean_amp_rel'][str(target)])
                            mean_list.append(self.dataset_mean[expname]['mean_features'][feature][str(target)])
                            std_list.append(self.dataset_mean[expname]['std_features'][feature][str(target)])

                    mean_array = numpy.array(mean_list)
                    std_array = numpy.array(std_list)
                    amp_rel_array = numpy.array(amp_rel_list)
                    #is_not_nan = ~numpy.isnan(mean_array)

                    figname = "features_" + expname
                    e = figs[figname]['axs'][fi].errorbar(amp_rel_list, mean_array,
                            yerr=std_array, marker='s', color='k', linewidth=1,
                            markersize=6, zorder=10, clip_on=False)
                    for b in e[1]:
                        b.set_clip_on(False)


        for i_fig, figname in enumerate(figs):
            fig = figs[figname]
            fig['fig'].savefig(fig['dirname'] + '/' + figname + '.pdf', dpi=300)
            plt.close(fig['fig'])


    def feature_config_all(self, version=None):

        #pickle.dump(self.dataset_mean, open(self.maindirname + "dataset_mean.pkl", 'w'), indent=4)
        self.create_feature_config(self.maindirname,
                            self.dataset_mean, version=version)


    def feature_config_cells(self, version=None):
        for i_cell, cellname in enumerate(self.dataset):
            dataset_cell_exp = self.dataset[cellname]['experiments']
            self.create_feature_config(self.maindirname+cellname+os.sep,
                            dataset_cell_exp, version=version)


    def analyse_threshold(self):

        logger.info(" Analysing threshold and hypamp and saving files to %s", self.maindirname)

        hyp_th = {}
        for cellname in self.thresholds_per_cell:
            hyp_th[cellname] = OrderedDict()
            hyp_th[cellname]['threshold'] = round(self.thresholds_per_cell[cellname],4)
            hyp_th[cellname]['hypamp'] = round(self.hypamps_per_cell[cellname],4)

        thresholds = [d for k, d in self.thresholds_per_cell.iteritems()]
        hypamps = [d for k, d in self.hypamps_per_cell.iteritems()]
        hyp_th['all'] = OrderedDict()
        hyp_th['all']['threshold'] = [round(numpy.mean(thresholds),4), round(numpy.std(thresholds),4)]
        hyp_th['all']['hypamp'] = [round(numpy.mean(hypamps),4), round(numpy.std(hypamps),4)]

        json.dump(hyp_th, open(self.maindirname + "hypamp_threshold.json", 'w'), indent=4)


    def create_feature_config(self, directory, dataset, version=None):

        logger.info(" Saving config files to %s", directory)

        stimulus_dict = OrderedDict()
        feature_dict = OrderedDict()

        if version == 'lnmc':

            stim = stimulus_dict
            feat = feature_dict

            for i_exp, expname in enumerate(self.experiments):
                if expname in dataset:
                    location = dataset[expname]['location']

                    for fi, feature in enumerate(self.features[expname]):
                        for it, target in enumerate(self.options["target"]):

                            if str(target) in dataset[expname]['mean_features'][feature]:

                                t = str(target)
                                stimname = expname + '_' + t

                                m = round(dataset[expname]['mean_features'][feature][str(target)],4)
                                s = round(dataset[expname]['std_features'][feature][str(target)],4)

                                if 'ncells' in dataset[expname]:
                                    n = int(dataset[expname]['ncells'][feature][str(target)])
                                else:
                                    n = 1

                                if s == 0: # prevent divison by 0
                                    s = 1e-3

                                if ~numpy.isnan(m):

                                    if stimname not in stim:
                                        stim[stimname] = OrderedDict()

                                    if stimname not in feat:
                                        feat[stimname] = OrderedDict()

                                    if location not in feat[stimname]:
                                        feat[stimname][location] = []

                                    feat[stimname][location].append(
                                                    OrderedDict([
                                                    ("feature",feature),
                                                    ("val",[m, s]),
                                                    ("n",n)
                                                    ]) )

                                    if expname in self.options["strict_stiminterval"].keys():
                                        strict_stiminterval = self.options["strict_stiminterval"][expname]
                                    else:
                                        strict_stiminterval = self.options["strict_stiminterval"]['base']
                                    feat[stimname][location][-1]["strict_stim"] = strict_stiminterval

                                    a = round(dataset[expname]['mean_amp'][str(target)],6)
                                    h = round(dataset[expname]['mean_hypamp'][str(target)],6)
                                    threshold = round(dataset[expname]['mean_amp_rel'][str(target)],4)
                                    ton = dataset[expname]['mean_ton']
                                    toff = dataset[expname]['mean_toff']
                                    tend = dataset[expname]['mean_tend']

                                    if 'stimuli' not in stim[stimname]:

                                        totduration = round(tend)
                                        delay = round(self.options["delay"] + ton)
                                        duration = round(toff-ton)

                                        if expname == 'H40S8': # special frequency pulse stimulus

                                            n = 8
                                            duration = 2.5
                                            stim[stimname]['type'] = 'StepProtocol'
                                            stim[stimname]['stimuli'] = OrderedDict()
                                            stim[stimname]['stimuli']['step'] = []
                                            totduration = delay + n * 25.
                                            threshold = 600. # threshold not estimated, used fixed values

                                            for s in range(n):
                                                stim[stimname]['stimuli']['step'].append(
                                                    OrderedDict([
                                                        ("delay", delay + s*25.),
                                                        ("amp", a),
                                                        ("thresh_perc", threshold),
                                                        ("duration", duration),
                                                        ("totduration", totduration),
                                                    ]))

                                            stim[stimname]['stimuli']['holding'] = OrderedDict([
                                                    ("delay", 0.0),
                                                    ("amp", h),
                                                    ("duration", totduration),
                                                    ("totduration", totduration),
                                                ])

                                        else:

                                            stim[stimname]['type'] = 'StepProtocol'
                                            stim[stimname]['stimuli'] = OrderedDict([
                                                            ('step',
                                                            OrderedDict([
                                                                ("delay", delay),
                                                                ("amp", a),
                                                                ("thresh_perc", threshold),
                                                                ("duration", duration),
                                                                ("totduration", totduration),
                                                            ])),
                                                            ('holding',
                                                            OrderedDict([
                                                                ("delay", 0.0),
                                                                ("amp", h),
                                                                ("duration", totduration),
                                                                ("totduration", totduration),
                                                            ])),
                                                        ])

        else:

            stim = OrderedDict()
            stimulus_dict[self.mainname] = stim

            feat = OrderedDict()
            feature_dict[self.mainname] = feat

            for i_exp, expname in enumerate(self.experiments):
                if expname in dataset:
                    location = dataset[expname]['location']

                    for fi, feature in enumerate(self.features[expname]):
                        for it, target in enumerate(self.options["target"]):
                            if str(target) in dataset[expname]['mean_features'][feature]:

                                stimname = expname + '_' + str(target)
                                if stimname not in stim:
                                    stim[stimname] = OrderedDict()

                                if stimname not in feat:
                                    feat[stimname] = OrderedDict()

                                m = dataset[expname]['mean_features'][feature][str(target)]
                                s = dataset[expname]['std_features'][feature][str(target)]

                                if ~numpy.isnan(m):
                                    if location not in feat[stimname]:
                                        feat[stimname][location] = OrderedDict()

                                    feat[stimname][location][feature] = [m, s]

                                a = round(dataset[expname]['mean_amp'][str(target)],6)
                                h = round(dataset[expname]['mean_hypamp'][str(target)],6)
                                ton = dataset[expname]['mean_ton']
                                toff = dataset[expname]['mean_toff']

                                if 'stimuli' not in stim[stimname]:

                                    totduration = round(self.options["delay"]+toff+self.options["posttime"])
                                    delay = round(self.options["delay"] + ton)
                                    duration = round(toff-ton)

                                    stim[stimname]['stimuli'] = [
                                                    OrderedDict([
                                                        ("delay", delay),
                                                        ("amp", a),
                                                        ("duration", duration),
                                                        ("totduration", totduration),
                                                    ]),
                                                    OrderedDict([
                                                        ("delay", 0.0),
                                                        ("amp", h),
                                                        ("duration", totduration),
                                                        ("totduration", totduration),
                                                    ]),
                                                ]

        #tools.print_dict(stimulus_dict)
        #tools.print_dict(feature_dict)
        feature_dict = self.clean_zero_std(feature_dict, directory)

        # clean protocols.json from amplitude for which no feature was computed
        all_stim_c = []
        for key, content in feature_dict.items():
            if key.startswith("ERROR"):
                continue
            else:
                for key_c in content:
                    if key_c not in all_stim_c:
                        all_stim_c.append(key_c)

        stim_dict_clean = OrderedDict()
        if not all_stim_c:
            stim_dict_clean = {"ERROR" : "No feature extracted. No protocol generated."}
        else:
            for key, content in stimulus_dict.items():
                for key_c in content:
                    if key_c in all_stim_c:
                        if key in stim_dict_clean:
                            stim_dict_clean[key][key_c] = stimulus_dict[key][key_c]
                        else:
                            stim_dict_clean[key] = OrderedDict()
                            stim_dict_clean[key][key_c] = stimulus_dict[key][key_c]

        # replacing stimulus dict with clean dict
        stimulus_dict = stim_dict_clean

        meta = OrderedDict()
        meta['version'] = self.githash

        s = json.dumps(meta, indent=2)
        s = tools.collapse_json(s, indent=6)
        with open(directory + "meta.json", "w") as f:
            f.write(s)

        s = json.dumps(stimulus_dict, indent=2)
        s = tools.collapse_json(s, indent=6)
        with open(directory + "protocols.json", "w") as f:
            f.write(s)

        s = json.dumps(feature_dict, indent=2)
        s = tools.collapse_json(s, indent=6)
        with open(directory + "features.json", "w") as f:
            f.write(s)


    def clean_zero_std(self, feature_dict, directory):
        # delete from feature_dict all empty dictionaries
        feat_final_dict = OrderedDict()
        for etype in feature_dict:
            for exp in feature_dict[etype]:
                for loc in feature_dict[etype][exp]:
                    crr_dict = OrderedDict()
                    for key_feat_val, feat_val in feature_dict[etype][exp][loc].items():
                        if feat_val[1] != 0.0:
                            crr_dict[key_feat_val] = feat_val
                    if crr_dict:
                        if etype not in feat_final_dict:
                            feat_final_dict[etype] = OrderedDict()
                        if exp not in feat_final_dict[etype]:
                            feat_final_dict[etype][exp] = OrderedDict()
                        feat_final_dict[etype][exp][loc] = crr_dict

        if not feat_final_dict:
            feat_final_dict["ERROR"] = "Either all features had zero standard deviation or no feature could be extracted from traces"
        return feat_final_dict
