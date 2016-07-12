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

try:
    import cPickle as pickle
except:
    import pickle

import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(stream=sys.stdout)
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

import tools
import plottools

class Extractor(object):

    def __init__(self, mainname='PC', config=collections.OrderedDict()):
        self.config = config

        self.path = config['path']
        self.cells = config['cells']
        self.features = config['features']
        self.format = config['format']

        self.dataset = collections.OrderedDict()
        self.dataset_mean = collections.OrderedDict()
        self.max_per_plot = 16
        self.amp_min = 0.01 # minimum current amplitude used for spike detection

        self.options = config['options']
        if "relative" not in self.options:
            self.options["relative"] = False

        if "tolerance" not in self.options:
            self.options["tolerance"] = 10

        if "target" not in self.options:
            self.options["target"] = [100., 150., 200., 250.]

        if "nanmean" not in self.options:
            self.options["nanmean"] = False

        if "delay" not in self.options:
            self.options["delay"] = 0

        if "posttime" not in self.options:
            self.options["posttime"] = 200

        self.colors = collections.OrderedDict()
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
        self.maindirname = "./" + mainname + "/"


    def create_dataset(self):

        logger.info(" Filling dataset")

        for i_cell, cellname in enumerate(self.cells):

            self.dataset[cellname] = collections.OrderedDict()
            self.dataset[cellname]['v_corr'] = self.cells[cellname]['v_corr']

            if 'ljp' in self.cells[cellname]:
                ljp = self.cells[cellname]['ljp']
            else:
                ljp = 0
            self.dataset[cellname]['ljp'] = ljp

            dataset_cell_exp = collections.OrderedDict()
            self.dataset[cellname]['experiments'] = dataset_cell_exp

            for i_exp, expname in enumerate(self.cells[cellname]['experiments']):

                logger.debug(" Adding experiment %s", expname)

                if expname not in self.experiments:
                    self.experiments.append(expname)

                dataset_cell_exp[expname] = collections.OrderedDict()

                dataset_cell_exp[expname]['location'] =\
                    self.cells[cellname]['experiments'][expname]['location']

                dataset_cell_exp[expname]['voltage'] = []
                dataset_cell_exp[expname]['current'] = []
                dataset_cell_exp[expname]['dt'] = []
                dataset_cell_exp[expname]['filename'] = []

                dataset_cell_exp[expname]['t'] = []
                dataset_cell_exp[expname]['ton'] = []
                dataset_cell_exp[expname]['toff'] = []
                dataset_cell_exp[expname]['amp'] = []
                dataset_cell_exp[expname]['hypamp'] = []

                files = self.cells[cellname]['experiments'][expname]['files']
                for i_file, filename in enumerate(files):

                    data = self.process_file(filename, cellname, expname)
                    dataset_cell_exp[expname]['voltage'].append(data['voltage'])
                    dataset_cell_exp[expname]['current'].append(data['current'])
                    dataset_cell_exp[expname]['dt'].append(data['dt'])
                    dataset_cell_exp[expname]['filename'].append(data['filename'])

                    dataset_cell_exp[expname]['t'].append(data['t'])
                    dataset_cell_exp[expname]['ton'].append(data['ton'])
                    dataset_cell_exp[expname]['toff'].append(data['toff'])
                    dataset_cell_exp[expname]['amp'].append(data['amp'])
                    dataset_cell_exp[expname]['hypamp'].append(data['hypamp'])


    def process_file(self, filename, cellname, expname):

        data = collections.OrderedDict()
        data['voltage'] = []
        data['current'] = []
        data['dt'] = []

        data['t'] = []
        data['ton'] = []
        data['toff'] = []
        data['amp'] = []
        data['hypamp'] = []
        data['filename'] = []

        ljp = self.dataset[cellname]['ljp']
        v_corr = self.dataset[cellname]['v_corr']

        if self.format == 'axon':

            if isinstance(filename, str) is False:
                raise Exception('Please provide a string with filename of axon file')

            logger.debug(" Adding axon file %s", filename)

            from neo import io
            f = self.path + cellname + '/' + filename + '.abf'
            r = io.AxonIO(filename = f) #
            bl = r.read_block(lazy=False, cascade=True)

            for i_seg, seg in enumerate(bl.segments):

                logger.debug(" Adding segment %d of %d", i_seg, len(bl.segments))

                voltage = numpy.array(seg.analogsignals[0]).astype(numpy.float64)
                current = numpy.array(seg.analogsignals[1]).astype(numpy.float64)

                dt = 1./int(seg.analogsignals[0].sampling_rate) * 1e3

                t = numpy.arange(len(voltage)) * dt

                # when does voltage change
                c_changes = numpy.where( abs(numpy.gradient(current, 1.)) > 0.0 )[0]

                # detect on and off of current
                c_changes2 = numpy.where( abs(numpy.gradient(c_changes, 1.)) > 10.0 )[0]

                ion = c_changes[c_changes2[0]]
                ioff = c_changes[-1]
                ton = ion * dt
                toff = ioff * dt

                # estimate hyperpolarization current
                hypamp = numpy.mean( current[0:ion] )

                # 10% distance to measure step current
                iborder = int((ioff-ion)*0.1)

                # depolarization amplitude
                amp = numpy.mean( current[ion+iborder:ioff-iborder] )

                voltage_dirty = voltage[:]

                # clean voltage from transients
                voltage[ion:ion+int(numpy.ceil(0.4/dt))] = voltage[ion+int(numpy.ceil(0.4/dt))]
                voltage[ioff:ioff+int(numpy.ceil(0.4/dt))] = voltage[ioff+int(numpy.ceil(0.4/dt))]

                # normalize membrane potential to known value (given in UCL excel sheet)
                if v_corr:
                    voltage = voltage - numpy.mean(voltage[0:ion]) + v_corr

                voltage = voltage - ljp

                # clip spikes after stimulus so they are not analysed
                voltage[ioff:] = numpy.clip(voltage[ioff:], -300, -40)

                if ('exclude' in self.cells[cellname] and
                    any(abs(self.cells[cellname]['exclude'] - amp) < 1e-4)):
                    logger.info(" Not using trace with amplitude %f", amp)

                else:
                    data['voltage'].append(voltage)
                    data['current'].append(current)
                    data['dt'].append(dt)

                    data['t'].append(t)
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
                        and any(abs(self.cells[cellname]['exclude'] - amp) < 1e-4)):
                        logger.info(" Not using trace with amplitude %f", amp)

                    else:
                        data['voltage'].append(voltage)
                        data['current'].append(current)
                        data['dt'].append(dt)

                        data['t'].append(t)
                        data['ton'].append(ton)
                        data['toff'].append(toff)
                        data['amp'].append(amp)
                        data['hypamp'].append(hypamp)
                        data['filename'].append(filename)

        return data


    def plt_traces(self):

        tools.makedir(self.maindirname)

        for i_cell, cellname in enumerate(self.dataset):

            dirname = self.maindirname+cellname
            tools.makedir(dirname)
            dataset_cell_exp = self.dataset[cellname]['experiments']

            for i_exp, expname in enumerate(dataset_cell_exp):

                voltages = dataset_cell_exp[expname]['voltage']
                amps = dataset_cell_exp[expname]['amp']
                ts = dataset_cell_exp[expname]['t']
                filenames = dataset_cell_exp[expname]['filename']

                voltages_flat = list(itertools.chain.from_iterable(voltages))
                amps_flat = list(itertools.chain.from_iterable(amps))
                ts_flat = list(itertools.chain.from_iterable(ts))
                filenames_flat = list(itertools.chain.from_iterable(filenames))
                colors_flat = []

                markercycler = cycle(self.markerlist)
                colorcycler = cycle(self.colorlist)

                for i_trace, trace in enumerate(voltages):
                    color = next(colorcycler)
                    for i_sig, sig in enumerate(trace):
                        colors_flat.append(color)

                isort = numpy.argsort(amps_flat)
                amps_flat = numpy.array(amps_flat)[isort]
                ts_flat = numpy.array(ts_flat)[isort]
                voltages_flat = numpy.array(voltages_flat)[isort]
                colors_flat = numpy.array(colors_flat)[isort]
                filenames_flat = numpy.array(filenames_flat)[isort]

                n_plot = len(voltages_flat)

                if n_plot <= self.max_per_plot:
                    frames = n_plot
                    n_fig = 1
                else:
                    frames = self.max_per_plot
                    n_fig = int(numpy.ceil(n_plot/float(self.max_per_plot)))

                axs = []
                figs = collections.OrderedDict()

                for i_fig in range(n_fig):
                    figname = cellname.split('/')[-1] + "_" + expname + "_" + str(i_fig)
                    axs = plottools.tiled_figure(figname, frames=frames, columns=2,
                                    figs=figs, axs=axs,
                                    top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.2)

                for i_plot in range(n_plot):
                    axs[i_plot].plot(ts_flat[i_plot], voltages_flat[i_plot], color=colors_flat[i_plot], clip_on=False)
                    axs[i_plot].set_title(cellname + " " + expname + " amp:" + str(amps_flat[i_plot]) + " file:" + filenames_flat[i_plot])

                for i_fig, figname in enumerate(figs):
                    fig = figs[figname]
                    fig['fig'].savefig(dirname + '/' + figname + '.pdf', dpi=300)
                    plt.close(fig['fig'])


    def extract_features(self):

        logger.info(" Extracting features")

        #efel.Settings.derivative_threshold = 5.0

        for i_cell, cellname in enumerate(self.dataset):

            dataset_cell_exp = self.dataset[cellname]['experiments']
            for i_exp, expname in enumerate(dataset_cell_exp):

                ts = dataset_cell_exp[expname]['t']
                voltages = dataset_cell_exp[expname]['voltage']
                #currents = dataset_cell_exp[expname]['current']
                tons = dataset_cell_exp[expname]['ton']
                toffs = dataset_cell_exp[expname]['toff']
                amps = dataset_cell_exp[expname]['amp']

                features_all = self.features[expname] + ['peak_time']

                if 'threshold' in self.cells[cellname]['experiments'][expname]:
                    threshold = self.cells[cellname]['experiments'][expname]['threshold']
                    logger.info(" Setting threshold to %f", threshold)
                    efel.setThreshold(threshold)

                dimensions = []
                for i_trace, trace in enumerate(voltages):
                    for i_sig, sig in enumerate(trace):
                        dimensions.append(i_trace)
                dataset_cell_exp[expname]['dimensions'] = dimensions

                ts = list(itertools.chain.from_iterable(ts))
                voltages = list(itertools.chain.from_iterable(voltages))
                #currents = list(itertools.chain.from_iterable(currents))
                tons = list(itertools.chain.from_iterable(tons))
                toffs = list(itertools.chain.from_iterable(toffs))
                amps = list(itertools.chain.from_iterable(amps))

                dataset_cell_exp[expname]['features'] = collections.OrderedDict()
                for feature in features_all:
                    dataset_cell_exp[expname]['features'][feature] = []
                dataset_cell_exp[expname]['features']['numspikes'] = []

                # iterate over all voltages individually to get features
                # and correct them if needed
                for i_seg in range(len(voltages)):

                    trace = collections.OrderedDict()
                    trace['T'] = ts[i_seg]
                    trace['V'] = voltages[i_seg]
                    trace['stim_start'] = [tons[i_seg]]
                    trace['stim_end'] = [toffs[i_seg]]
                    traces = [trace]
                    amp = amps[i_seg]

                    fel_vals = efel.getFeatureValues(traces, features_all, raise_warnings=False)

                    for feature in features_all:

                        if feature == 'peak_time':
                            peak_times = fel_vals[0][feature]
                            if (len(peak_times) > 0) and amp >= self.amp_min:
                                numspike = len(peak_times)
                            else:
                                numspike = 0

                        if fel_vals[0][feature] is not None:
                            f = numpy.mean( fel_vals[0][feature] )
                        else:
                            f = float('nan')

                        dataset_cell_exp[expname]['features'][feature].append(f)

                    dataset_cell_exp[expname]['features']['numspikes'].append(numspike)


    def mean_features(self):

        logger.info(" Calculating mean features")

        # mean for each cell
        for i_cell, cellname in enumerate(self.dataset):

            dataset_cell_exp = self.dataset[cellname]['experiments']
            for i_exp, expname in enumerate(dataset_cell_exp):

                # define
                dataset_cell_exp[expname]['mean_amp'] = collections.OrderedDict()
                dataset_cell_exp[expname]['mean_amp_rel'] = collections.OrderedDict()

                dataset_cell_exp[expname]['mean_features'] = collections.OrderedDict()
                dataset_cell_exp[expname]['std_features'] = collections.OrderedDict()

                for target in self.options["target"]:
                    dataset_cell_exp[expname]['mean_amp'][str(target)] = []
                    dataset_cell_exp[expname]['mean_amp_rel'][str(target)] = []

                for feature in self.features[expname]:
                    dataset_cell_exp[expname]['mean_features'][feature] = collections.OrderedDict()
                    dataset_cell_exp[expname]['std_features'][feature] = collections.OrderedDict()

                hypamp = dataset_cell_exp[expname]['hypamp']
                hypamp = numpy.mean(numpy.array(hypamp))

                ton = dataset_cell_exp[expname]['ton']
                ton = numpy.mean(numpy.array(ton))

                toff = dataset_cell_exp[expname]['toff']
                toff = numpy.mean(numpy.array(toff))

                amp = dataset_cell_exp[expname]['amp']
                numspikes = dataset_cell_exp[expname]['features']['numspikes']
                feature_array = dataset_cell_exp[expname]['features']

                amp = numpy.array(list(itertools.chain.from_iterable(amp)))

                amp_threshold, amp_rel = self.get_threshold(amp, numspikes)
                logger.info(" %s threshold amplitude: %f hypamp: %f", cellname, amp_threshold, hypamp)

                # save amplitude results
                for target in self.options["target"]:
                    idx = (amp_rel >= target - self.options["tolerance"]) &\
                            (amp_rel <= target + self.options["tolerance"])

                    amp_target = numpy.array(amp)[idx]
                    meanamp_target = numpy.mean(amp_target) # nanmean
                    dataset_cell_exp[expname]['mean_amp'][str(target)] = meanamp_target

                    # equal to amp_target if amplitude not measured relative to threshold
                    amp_rel_target = numpy.array(amp_rel)[idx]
                    meanamp_rel_target = numpy.mean(amp_rel_target) # nanmean
                    dataset_cell_exp[expname]['mean_amp_rel'][str(target)] = meanamp_rel_target

                dataset_cell_exp[expname]['mean_hypamp'] = hypamp
                dataset_cell_exp[expname]['mean_ton'] = ton
                dataset_cell_exp[expname]['mean_toff'] = toff

                for fi, feature in enumerate(self.features[expname]):
                    feat_vals = numpy.array(feature_array[feature])
                    for ti, target in enumerate(self.options["target"]):
                        idx = (amp_rel >= target - self.options["tolerance"]) &\
                                (amp_rel <= target + self.options["tolerance"])
                        feat = numpy.array(feat_vals)[idx]

                        if self.options["nanmean"]:
                            meanfeat = numpy.nanmean(feat)
                            stdfeat = numpy.nanstd(feat)
                        else:
                            meanfeat = numpy.mean(feat)
                            stdfeat = numpy.std(feat)
                        dataset_cell_exp[expname]['mean_features'][feature][str(target)] = meanfeat
                        dataset_cell_exp[expname]['std_features'][feature][str(target)] = stdfeat


        # mean for all cells
        for i_exp, expname in enumerate(self.experiments):

            #collect everything in global structure
            self.dataset_mean[expname] = collections.OrderedDict()
            self.dataset_mean[expname]['amp'] = collections.OrderedDict()
            self.dataset_mean[expname]['amp_rel'] = collections.OrderedDict()
            self.dataset_mean[expname]['hypamp'] = []
            self.dataset_mean[expname]['ton'] = []
            self.dataset_mean[expname]['toff'] = []
            self.dataset_mean[expname]['features'] = collections.OrderedDict()

            for feature in self.features[expname]:
                self.dataset_mean[expname]['features'][feature] = collections.OrderedDict()
                for target in self.options["target"]:
                    self.dataset_mean[expname]['features'][feature][str(target)] = []

            for target in self.options["target"]:
                self.dataset_mean[expname]['amp'][str(target)] = []
                self.dataset_mean[expname]['amp_rel'][str(target)] = []

            for i_cell, cellname in enumerate(self.dataset):

                print cellname
                #tools.print_dict(self.dataset[cellname])
                dataset_cell_exp = self.dataset[cellname]['experiments']

                self.dataset_mean[expname]['location'] =\
                        dataset_cell_exp[expname]['location']

                hypamp = dataset_cell_exp[expname]['mean_hypamp']
                self.dataset_mean[expname]['hypamp'].append(hypamp)

                ton = dataset_cell_exp[expname]['mean_ton']
                self.dataset_mean[expname]['ton'].append(ton)

                toff = dataset_cell_exp[expname]['mean_toff']
                self.dataset_mean[expname]['toff'].append(toff)

                for target in self.options["target"]:
                    amp = dataset_cell_exp[expname]['mean_amp'][str(target)]
                    self.dataset_mean[expname]['amp'][str(target)].append(amp)
                    amp_rel = dataset_cell_exp[expname]['mean_amp_rel'][str(target)]
                    self.dataset_mean[expname]['amp_rel'][str(target)].append(amp_rel)

                for feature in self.features[expname]:
                    for target in self.options["target"]:
                        result = dataset_cell_exp[expname]['mean_features'][feature][str(target)]
                        self.dataset_mean[expname]['features'][feature][str(target)].append(result)

            #create means
            self.dataset_mean[expname]['mean_amp'] = collections.OrderedDict()
            self.dataset_mean[expname]['mean_amp_rel'] = collections.OrderedDict()
            self.dataset_mean[expname]['mean_features'] = collections.OrderedDict()
            self.dataset_mean[expname]['std_features'] = collections.OrderedDict()

            for feature in self.features[expname]:
                self.dataset_mean[expname]['mean_features'][feature] = collections.OrderedDict()
                self.dataset_mean[expname]['std_features'][feature] = collections.OrderedDict()

            hypamp = self.dataset_mean[expname]['hypamp']
            self.dataset_mean[expname]['mean_hypamp'] = numpy.mean(hypamp)

            ton = self.dataset_mean[expname]['ton']
            self.dataset_mean[expname]['mean_ton'] = numpy.mean(ton)

            toff = self.dataset_mean[expname]['toff']
            self.dataset_mean[expname]['mean_toff'] = numpy.mean(toff)

            for target in self.options["target"]:
                amp = self.dataset_mean[expname]['amp'][str(target)]
                self.dataset_mean[expname]['mean_amp'][str(target)] = numpy.mean(amp)

                amp_rel = self.dataset_mean[expname]['amp_rel'][str(target)]
                self.dataset_mean[expname]['mean_amp_rel'][str(target)] = numpy.mean(amp)

            for feature in self.features[expname]:
                for target in self.options["target"]:
                    feat = self.dataset_mean[expname]['features'][feature][str(target)]

                    if self.options["nanmean"]:
                        self.dataset_mean[expname]['mean_features'][feature][str(target)] = numpy.nanmean(feat)
                        self.dataset_mean[expname]['std_features'][feature][str(target)] = numpy.nanstd(feat)
                    else:
                        self.dataset_mean[expname]['mean_features'][feature][str(target)] = numpy.mean(feat)
                        self.dataset_mean[expname]['std_features'][feature][str(target)] = numpy.std(feat)


    def get_threshold(self, amp, numspikes):

        isort = numpy.argsort(amp)
        amps_sort = numpy.array(amp)[isort]
        numspikes_sort = numpy.array(numspikes)[isort]
        i_threshold = numpy.where(numspikes_sort>=1)[0][0]
        amp_threshold = amps_sort[i_threshold]

        if self.options["relative"]:
            amp_rel = amp/amp_threshold * 100.
        else:
            amp_rel = amp

        return amp_threshold, amp_rel


    def plt_features(self):

        logger.info(" Plotting features")

        figs = collections.OrderedDict()

        tools.makedir(self.maindirname)

        markercyclercell = cycle(self.markerlist)
        colorcyclercell = cycle(self.colorlist)

        for i_cell, cellname in enumerate(self.dataset):

            dirname = self.maindirname+cellname
            tools.makedir(dirname)

            colorcell = next(colorcyclercell)
            markercell = next(markercyclercell)
            dataset_cell_exp = self.dataset[cellname]['experiments']

            for i_exp, expname in enumerate(dataset_cell_exp):

                if (i_cell == 0):
                    figname = "features_" + expname
                    plottools.tiled_figure(figname, frames=len(self.features[expname]),
                                    columns=3, figs=figs, dirname=self.maindirname,
                                    top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.3)


                figname = "features_" + cellname.split('/')[-1] + "_" + expname
                axs_cell = plottools.tiled_figure(figname, frames=len(self.features[expname]),
                                    columns=3, figs=figs, dirname=dirname,
                                    top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.3)


                hypamp = numpy.mean(numpy.array(dataset_cell_exp[expname]['hypamp']))
                amp = dataset_cell_exp[expname]['amp']
                numspikes = dataset_cell_exp[expname]['features']['numspikes']
                feature_array = dataset_cell_exp[expname]['features']

                amp = numpy.array(list(itertools.chain.from_iterable(amp)))
                amp_threshold, amp_vals = self.get_threshold(amp, numspikes)

                for fi, feature in enumerate(self.features[expname]):

                    dimensions = dataset_cell_exp[expname]['dimensions']
                    dimensions = numpy.concatenate(([0], numpy.where(numpy.diff(dimensions)>0)[0]+1, [len(dimensions)]))

                    feat_vals = numpy.array(feature_array[feature])

                    markercycler = cycle(self.markerlist)
                    colorcycler = cycle(self.colorlist)

                    for i in range(len(dimensions)-1):

                        color = next(colorcycler)
                        marker = next(markercycler)

                        istart = dimensions[i]
                        iend = dimensions[i+1]
                        feat_vals0 = feat_vals[istart:iend]
                        amp_vals0 = amp_vals[istart:iend]
                        is_not_nan = ~numpy.isnan(feat_vals0)

                        axs_cell[fi].plot(amp_vals0[is_not_nan], feat_vals0[is_not_nan], "",
                                    marker=marker, color=color, markersize=5, zorder=1,
                                    linewidth=1, markeredgecolor = 'none', clip_on=False)
                        axs_cell[fi].set_xticks(self.options["target"])
                        #axs_cell[fi].set_xticklabels(())
                        axs_cell[fi].set_title(feature)

                        figname = "features_" + expname
                        figs[figname]['axs'][fi].plot(amp_vals0[is_not_nan], feat_vals0[is_not_nan], "",
                                    marker=markercell, color=colorcell, markersize=5, zorder=1,
                                    linewidth=1, markeredgecolor = 'none', clip_on=False)
                        figs[figname]['axs'][fi].set_xticks(self.options["target"])
                        figs[figname]['axs'][fi].set_title(feature)

                    if 'mean_features' in dataset_cell_exp[expname]:

                        amp_rel_list = []
                        mean_list = []
                        std_list = []
                        for target in self.options["target"]:
                            amp_rel_list.append(dataset_cell_exp[expname]['mean_amp_rel'][str(target)])
                            mean_list.append(dataset_cell_exp[expname]['mean_features'][feature][str(target)])
                            std_list.append(dataset_cell_exp[expname]['std_features'][feature][str(target)])

                        mean_array = numpy.array(mean_list)
                        std_array = numpy.array(std_list)
                        amp_rel_array = numpy.array(amp_rel_list)

                        e = axs_cell[fi].errorbar(amp_rel_array, mean_array, yerr=std_array, marker='s', color='k',
                                    linewidth=1, markersize=6, zorder=10, clip_on = False)
                        axs_cell[fi].set_xticks(self.options["target"])
                        for b in e[1]:
                            b.set_clip_on(False)


        for i_exp, expname in enumerate(self.experiments):

            if expname in self.dataset_mean:
                for fi, feature in enumerate(self.features[expname]):
                    amp_rel_list = []
                    mean_list = []
                    std_list = []
                    for target in self.options["target"]:
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


    def feature_config_all(self):
        self.create_feature_config(self.maindirname, self.dataset_mean)


    def feature_config_cells(self):
        for i_cell, cellname in enumerate(self.dataset):
            dataset_cell_exp = self.dataset[cellname]['experiments']
            self.create_feature_config(self.maindirname+cellname+'/', dataset_cell_exp)


    def create_feature_config(self, directory, dataset):

        import json
        logger.info(" Saving config files to %s", directory)

        stimulus_dict = collections.OrderedDict()
        feature_dict = collections.OrderedDict()

        stim = collections.OrderedDict()
        stimulus_dict[self.mainname] = stim

        feat = collections.OrderedDict()
        feature_dict[self.mainname] = feat

        for i_exp, expname in enumerate(self.experiments):
            if expname in dataset:
                location = dataset[expname]['location']

                for fi, feature in enumerate(self.features[expname]):
                    for it, target in enumerate(self.options["target"]):

                        stimname = expname + '_' + str(target)
                        if stimname not in stim:
                            stim[stimname] = collections.OrderedDict()

                        if stimname not in feat:
                            feat[stimname] = collections.OrderedDict()

                        m = dataset[expname]['mean_features'][feature][str(target)]
                        s = dataset[expname]['std_features'][feature][str(target)]

                        if ~numpy.isnan(m):
                            if location not in feat[stimname]:
                                feat[stimname][location] = collections.OrderedDict()

                            feat[stimname][location][feature] = [m, s]

                        a = round(dataset[expname]['mean_amp'][str(target)],6)
                        h = round(dataset[expname]['mean_hypamp'],6)
                        ton = dataset[expname]['mean_ton']
                        toff = dataset[expname]['mean_toff']

                        if 'stimuli' not in stim[stimname]:

                            totduration = round(self.options["delay"]+toff+self.options["posttime"])
                            delay = round(self.options["delay"] + ton)
                            duration = round(toff-ton)

                            stim[stimname]['stimuli'] = [
                                            collections.OrderedDict([
                                                ("delay", delay),
                                                ("amp", a),
                                                ("duration", duration),
                                                ("totduration", totduration),
                                            ]),
                                            collections.OrderedDict([
                                                ("delay", 0.0),
                                                ("amp", h),
                                                ("duration", totduration),
                                                ("totduration", totduration),
                                            ]),
                                        ]

        #tools.print_dict(stimulus_dict)
        #tools.print_dict(feature_dict)
        json.dump(stimulus_dict, open(directory + "protocols.json", 'w'), indent=4)
        json.dump(feature_dict, open(directory + "features.json", 'w'), indent=4)
