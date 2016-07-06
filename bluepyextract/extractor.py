from neo import io
import matplotlib
matplotlib.use('Agg', warn=True)
import matplotlib.pyplot as plt
import numpy
import sys
import efel
import igorpy
import os
import fnmatch
import itertools
import collections
from itertools import cycle

try:
    import cPickle as pickle
except:
    import pickle

import tools
import plottools

class Extractor(object):

    def __init__(self, mainname='PC'):
        self.cells = {}
        self.path = ""
        self.dataset = collections.OrderedDict()
        self.dataset_processed = collections.OrderedDict()
        self.filetype = 'abf'
        self.features = []
        self.max_per_plot = 16
        self.amp_min = 0.01 # minimum current amplitude used for spike detection
        self.relative = True
        self.tolerance = 10
        self.target_normap = [100., 150., 200., 250.]

        self.colors = {}
        self.colors['b1'] = '#1F78B4' #377EB8
        self.colors['b2']= '#A6CEE3'
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

        self.maindirname = "./" + mainname + "/"

        self.stimulus_string = ""
        self.feature_string = ""
        self.use_nanmean = False


    def get_threshold(self, amp, numspikes):

        isort = numpy.argsort(amp)
        amps_sort = numpy.array(amp)[isort]
        numspikes_sort = numpy.array(numspikes)[isort]
        i_threshold = numpy.where(numspikes_sort>=1)[0][0]
        amp_threshold = amps_sort[i_threshold]

        if self.relative:
            amp_rel = amp/amp_threshold * 100
        else:
            amp_rel = amp

        return amp_threshold, amp_rel


    def create_dataset(self):

        print "- filling dataset"

        for i_cell, cellname in enumerate(self.cells):

            self.dataset[cellname] = {}
            self.dataset[cellname]['v_corr'] = self.cells[cellname]['v_corr']
            self.dataset[cellname]['experiments'] = {}
            dataset_exp = self.dataset[cellname]['experiments']

            for i_exp, expname in enumerate(self.cells[cellname]['experiments']):

                if expname not in self.experiments:
                    self.experiments.append(expname)

                dataset_exp[expname] = {}
                dataset_exp[expname]['voltage'] = []
                dataset_exp[expname]['current'] = []
                dataset_exp[expname]['dt'] = []
                dataset_exp[expname]['tracename'] = []

                for i_trace, tracename in enumerate(self.cells[cellname]['experiments'][expname]['traces']):

                    dataset_exp[expname]['voltage'].append([])
                    dataset_exp[expname]['current'].append([])
                    dataset_exp[expname]['dt'].append([])
                    dataset_exp[expname]['tracename'].append([])

                    if self.filetype == 'abf':

                        f = self.path + cellname + '/' + tracename + '.' + self.filetype
                        r = io.AxonIO(filename = f) #
                        bl = r.read_block(lazy=False, cascade=True)

                        for i_seg, seg in enumerate(bl.segments):

                            print 'cell:', cellname, 'experiment:', expname, 'i_trace:', i_trace, "Segment", i_seg
                            voltage = numpy.array(seg.analogsignals[0])
                            current = numpy.array(seg.analogsignals[1])
                            dt = 1./int(seg.analogsignals[0].sampling_rate) * 1e3
                            dataset_exp[expname]['voltage'][-1].append(voltage)
                            dataset_exp[expname]['current'][-1].append(current)
                            dataset_exp[expname]['dt'][-1].append(dt)
                            dataset_exp[expname]['tracename'][-1].append(tracename)

                    elif self.filetype == 'LCCR':

                        import csv
                        filename = self.path + '/' + tracename + '.txt'

                        dt = self.cells[cellname]['experiments'][expname]['dt']
                        amplitudes = self.cells[cellname]['experiments'][expname]['amplitudes']

                        with open(filename, 'rb') as f:
                            reader = csv.reader(f, delimiter='\t')
                            columns = zip(*reader)
                            length = numpy.shape(columns)[1]

                            for ic, column in enumerate(columns):

                                voltage = numpy.zeros(length)
                                for istr, string in enumerate(column):
                                    if (string != "-") and (string != ""):
                                        voltage[istr] = float(string)

                                dataset_exp[expname]['voltage'][-1].append(voltage)
                                dataset_exp[expname]['tracename'][-1].append(tracename)


    def preprocess_dataset(self):

        print "- preprocessing dataset"

        for i_cell, cellname in enumerate(self.dataset):

            v_corr = self.dataset[cellname]['v_corr']

            if 'ljp' in self.cells[cellname]:
                ljp = self.cells[cellname]['ljp']
                print "- ljp:", ljp
            else:
                ljp = 0

            self.dataset[cellname]['experiments_preprocessed'] = {}

            dataset_exp = self.dataset[cellname]['experiments']
            dataset_exp_pre = self.dataset[cellname]['experiments_preprocessed']

            for i_exp, expname in enumerate(dataset_exp):

                dataset_exp_pre[expname] = {}

                if self.filetype == 'abf':

                    voltages = dataset_exp[expname]['voltage']
                    currents = dataset_exp[expname]['current']
                    dts = dataset_exp[expname]['dt']
                    tracenames = dataset_exp[expname]['tracename']

                    dataset_exp_pre[expname]['voltage'] = []
                    dataset_exp_pre[expname]['current'] = []
                    dataset_exp_pre[expname]['dt'] = []
                    dataset_exp_pre[expname]['tracename'] = []

                    dataset_exp_pre[expname]['t'] = []
                    dataset_exp_pre[expname]['ton'] = []
                    dataset_exp_pre[expname]['toff'] = []
                    dataset_exp_pre[expname]['amp'] = []
                    dataset_exp_pre[expname]['hypamp'] = []

                    for i_trace in range(len(voltages)):

                        dataset_exp_pre[expname]['voltage'].append([])
                        dataset_exp_pre[expname]['current'].append([])
                        dataset_exp_pre[expname]['dt'].append([])
                        dataset_exp_pre[expname]['tracename'].append([])

                        dataset_exp_pre[expname]['t'].append([])
                        dataset_exp_pre[expname]['ton'].append([])
                        dataset_exp_pre[expname]['toff'].append([])
                        dataset_exp_pre[expname]['amp'].append([])
                        dataset_exp_pre[expname]['hypamp'].append([])

                        for i_seg in range(len(voltages[i_trace])):
                            voltage = voltages[i_trace][i_seg]
                            current = currents[i_trace][i_seg]
                            dt = dts[i_trace][i_seg]
                            tracename = tracenames[i_trace][i_seg]

                            t = numpy.arange(len(voltage)) * dt

                            c_changes = numpy.where( abs(numpy.gradient(current, 1.)) > 0.0 )[0]  # when does voltage change
                            c_changes2 = numpy.where( abs(numpy.gradient(c_changes, 1.)) > 10.0 )[0] # detect on and off of current

                            ion = c_changes[c_changes2[0]]
                            ioff = c_changes[-1]
                            ton = ion * dt
                            toff = ioff * dt

                            hypamp = numpy.mean( current[0:ion] )  # estimate hyperpolarization current
                            iborder = int((ioff-ion)*0.1)  # 10% distance to measure step current
                            amp = numpy.mean( current[ion+iborder:ioff-iborder] )  # depolarization amplitude
                            #print "hypamp: ", hypamp, "amp:", amp, "ton: ", ton, "toff:", toff

                            voltage_dirty = voltage[:]

                            # clean voltage from transients
                            voltage[ion:ion+numpy.ceil(0.4/dt)] = voltage[ion+numpy.ceil(0.4/dt)]
                            voltage[ioff:ioff+numpy.ceil(0.4/dt)] = voltage[ioff+numpy.ceil(0.4/dt)]

                            if v_corr:
                                voltage = voltage - numpy.mean(voltage[0:ion]) + v_corr # normalize membrane potential to value given in excel sheet

                            voltage = voltage - ljp

                            voltage[ioff:] = numpy.clip(voltage[ioff:], -300, -40) # clip spikes after stimulus so they are not analysed

                            if 'ex_amp' in self.cells[cellname] and any(abs(self.cells[cellname]['ex_amp'] - amp) < 1e-4):
                                print "- not using trace with amp:", amp

                            else:

                                dataset_exp_pre[expname]['voltage'][-1].append(voltage)
                                dataset_exp_pre[expname]['current'][-1].append(current)
                                dataset_exp_pre[expname]['dt'][-1].append(dt)
                                dataset_exp_pre[expname]['tracename'][-1].append(tracename)

                                dataset_exp_pre[expname]['t'][-1].append(t)
                                dataset_exp_pre[expname]['ton'][-1].append(ton)
                                dataset_exp_pre[expname]['toff'][-1].append(toff)
                                dataset_exp_pre[expname]['amp'][-1].append(amp)
                                dataset_exp_pre[expname]['hypamp'][-1].append(hypamp)


                elif self.filetype == 'LCCR':

                    voltages = dataset_exp[expname]['voltage']
                    tracenames = dataset_exp[expname]['tracename']

                    dt = self.cells[cellname]['experiments'][expname]['dt']
                    amplitudes = self.cells[cellname]['experiments'][expname]['amplitudes']
                    hypamp = self.cells[cellname]['experiments'][expname]['hypamp']
                    ton = self.cells[cellname]['experiments'][expname]['startstop'][0]
                    toff = self.cells[cellname]['experiments'][expname]['startstop'][1]

                    dataset_exp_pre[expname]['voltage'] = []
                    dataset_exp_pre[expname]['dt'] = []
                    dataset_exp_pre[expname]['tracename'] = []

                    dataset_exp_pre[expname]['t'] = []
                    dataset_exp_pre[expname]['ton'] = []
                    dataset_exp_pre[expname]['toff'] = []
                    dataset_exp_pre[expname]['amp'] = []
                    dataset_exp_pre[expname]['hypamp'] = []

                    for i_trace in range(len(voltages)):

                        dataset_exp_pre[expname]['voltage'].append([])
                        dataset_exp_pre[expname]['dt'].append([])
                        dataset_exp_pre[expname]['tracename'].append([])

                        dataset_exp_pre[expname]['t'].append([])
                        dataset_exp_pre[expname]['ton'].append([])
                        dataset_exp_pre[expname]['toff'].append([])
                        dataset_exp_pre[expname]['amp'].append([])
                        dataset_exp_pre[expname]['hypamp'].append([])

                        for i_seg in range(len(voltages[i_trace])):
                            voltage = voltages[i_trace][i_seg]
                            tracename = tracenames[i_trace][i_seg]

                            t = numpy.arange(len(voltage)) * dt
                            amp = amplitudes[i_seg]

                            if 'ex_amp' in self.cells[cellname] and any(abs(self.cells[cellname]['ex_amp'] - amp) < 1e-4):
                                print "- not using trace with amp:", amp

                            else:

                                voltage = voltage - ljp # correct liquid junction potential

                                # remove last 100 ms
                                voltage = voltage[0:int(-100./dt)]
                                t = t[0:int(-100./dt)]

                                dataset_exp_pre[expname]['voltage'][-1].append(voltage)
                                dataset_exp_pre[expname]['dt'][-1].append(dt)
                                dataset_exp_pre[expname]['tracename'][-1].append(tracename)

                                dataset_exp_pre[expname]['t'][-1].append(t)
                                dataset_exp_pre[expname]['ton'][-1].append(ton)
                                dataset_exp_pre[expname]['toff'][-1].append(toff)
                                dataset_exp_pre[expname]['amp'][-1].append(amp)
                                dataset_exp_pre[expname]['hypamp'][-1].append(hypamp)


    def plt_traces(self):

        tools.makedir(self.maindirname)

        for i_cell, cellname in enumerate(self.dataset):

            dirname = self.maindirname+cellname
            tools.makedir(dirname)
            dataset_exp_pre = self.dataset[cellname]['experiments_preprocessed']

            for i_exp, expname in enumerate(dataset_exp_pre):

                voltages = dataset_exp_pre[expname]['voltage']
                amps = dataset_exp_pre[expname]['amp']
                ts = dataset_exp_pre[expname]['t']
                tracenames = dataset_exp_pre[expname]['tracename']

                voltages_flat = list(itertools.chain.from_iterable(voltages))
                amps_flat = list(itertools.chain.from_iterable(amps))
                ts_flat = list(itertools.chain.from_iterable(ts))
                tracenames_flat = list(itertools.chain.from_iterable(tracenames))
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
                tracenames_flat = numpy.array(tracenames_flat)[isort]

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
                    axs = plottools.tiled_figure(figname, frames=frames, columns=2, figs=figs, axs=axs,
                                            top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.3)

                for i_plot in range(n_plot):
                    axs[i_plot].plot(ts_flat[i_plot], voltages_flat[i_plot], color=colors_flat[i_plot], clip_on=False)
                    axs[i_plot].set_title(cellname + " " + expname + " amp:" + str(amps_flat[i_plot]) + " file:" + tracenames_flat[i_plot])

                for i_fig, figname in enumerate(figs):
                    fig = figs[figname]
                    fig['fig'].savefig(dirname + '/' + figname + '.pdf', dpi=300)
                    plt.close(fig['fig'])


    def extract_features(self, features):

        print "- extracting features"

        self.features = features
        self.features_all = self.features + ['peak_time']

        #efel.Settings.derivative_threshold = 5.0

        for i_cell, cellname in enumerate(self.dataset):

            dataset_exp_pre = self.dataset[cellname]['experiments_preprocessed']
            for i_exp, expname in enumerate(dataset_exp_pre):

                ts = dataset_exp_pre[expname]['t']
                voltages = dataset_exp_pre[expname]['voltage']
                #currents = dataset_exp_pre[expname]['current']
                tons = dataset_exp_pre[expname]['ton']
                toffs = dataset_exp_pre[expname]['toff']
                amps = dataset_exp_pre[expname]['amp']

                if 'threshold' in self.cells[cellname]['experiments'][expname]:
                    threshold = self.cells[cellname]['experiments'][expname]['threshold']
                    print "- Setting threshold to", threshold
                    efel.setThreshold(threshold)

                dimensions = []
                for i_trace, trace in enumerate(voltages):
                    for i_sig, sig in enumerate(trace):
                        dimensions.append(i_trace)
                dataset_exp_pre[expname]['dimensions'] = dimensions

                ts = list(itertools.chain.from_iterable(ts))
                voltages = list(itertools.chain.from_iterable(voltages))
                #currents = list(itertools.chain.from_iterable(currents))
                tons = list(itertools.chain.from_iterable(tons))
                toffs = list(itertools.chain.from_iterable(toffs))
                amps = list(itertools.chain.from_iterable(amps))

                dataset_exp_pre[expname]['features'] = {}
                for feature in self.features_all:
                    dataset_exp_pre[expname]['features'][feature] = []
                dataset_exp_pre[expname]['features']['numspikes'] = []

                for i_seg in range(len(voltages)):

                    trace = {}
                    trace['T'] = ts[i_seg]
                    trace['V'] = voltages[i_seg]
                    trace['stim_start'] = [tons[i_seg]]
                    trace['stim_end'] = [toffs[i_seg]]
                    traces = [trace]
                    amp = amps[i_seg]

                    fel_vals = efel.getFeatureValues(traces, self.features_all)

                    for feature in self.features_all:

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

                        dataset_exp_pre[expname]['features'][feature].append(f)

                    dataset_exp_pre[expname]['features']['numspikes'].append(numspike)


    def mean_features(self):

        print "- calculating mean features"

        # mean for each cell
        for i_cell, cellname in enumerate(self.dataset):

            dataset_exp_pre = self.dataset[cellname]['experiments_preprocessed']
            for i_exp, expname in enumerate(dataset_exp_pre):

                # define
                dataset_exp_pre[expname]['mean_amp'] = {}
                dataset_exp_pre[expname]['mean_amp_rel'] = {}

                #dataset_exp_pre[expname]['mean_hypamp'] = []
                #dataset_exp_pre[expname]['mean_ton'] = []
                #dataset_exp_pre[expname]['mean_toff'] = []

                dataset_exp_pre[expname]['mean_features'] = {}
                dataset_exp_pre[expname]['std_features'] = {}

                for target in self.target_normap:
                    dataset_exp_pre[expname]['mean_amp'][str(target)] = []
                    dataset_exp_pre[expname]['mean_amp_rel'][str(target)] = []

                for feature in self.features:
                    dataset_exp_pre[expname]['mean_features'][feature] = {}
                    dataset_exp_pre[expname]['std_features'][feature] = {}
                    #for target in self.target_normap:
                    #    dataset_exp_pre[expname]['mean_features'][feature][str(target)] = []
                    #    dataset_exp_pre[expname]['std_features'][feature][str(target)] = []

                hypamp = dataset_exp_pre[expname]['hypamp']
                hypamp = numpy.mean(numpy.array(hypamp))

                ton = dataset_exp_pre[expname]['ton']
                ton = numpy.mean(numpy.array(ton))

                toff = dataset_exp_pre[expname]['toff']
                toff = numpy.mean(numpy.array(toff))

                amp = dataset_exp_pre[expname]['amp']
                numspikes = dataset_exp_pre[expname]['features']['numspikes']
                feature_array = dataset_exp_pre[expname]['features']

                amp = numpy.array(list(itertools.chain.from_iterable(amp)))

                amp_threshold, amp_rel = self.get_threshold(amp, numspikes)
                print cellname, "threshold amplitude:", amp_threshold, "hypamp:", hypamp

                # save amplitude results
                for target in self.target_normap:
                    amp_target = numpy.array(amp)[(amp_rel >= target - self.tolerance) & (amp_rel <= target + self.tolerance)]
                    meanamp_target = numpy.mean(amp_target) # nanmean
                    #dataset_exp_pre[expname]['mean_amp'][str(target)].append(meanamp_target)
                    dataset_exp_pre[expname]['mean_amp'][str(target)] = meanamp_target

                    amp_rel_target = numpy.array(amp_rel)[(amp_rel >= target - self.tolerance) & (amp_rel <= target + self.tolerance)]
                    meanamp_rel_target = numpy.mean(amp_rel_target) # nanmean
                    #dataset_exp_pre[expname]['mean_amp_rel'][str(target)].append(meanamp_rel_target)
                    dataset_exp_pre[expname]['mean_amp_rel'][str(target)] = meanamp_rel_target

                #dataset_exp_pre[expname]['mean_hypamp'].append(hypamp)
                #dataset_exp_pre[expname]['mean_ton'].append(ton)
                #dataset_exp_pre[expname]['mean_toff'].append(toff)

                dataset_exp_pre[expname]['mean_hypamp'] = hypamp
                dataset_exp_pre[expname]['mean_ton'] = ton
                dataset_exp_pre[expname]['mean_toff'] = toff

                for fi, feature in enumerate(self.features):
                    feat_vals = numpy.array(feature_array[feature])
                    for ti, target in enumerate(self.target_normap):
                        feat = numpy.array(feat_vals)[(amp_rel >= target - self.tolerance) & (amp_rel <= target + self.tolerance)]
                        if self.use_nanmean:
                            meanfeat = numpy.nanmean(feat)
                            stdfeat = numpy.nanstd(feat)
                        else:
                            meanfeat = numpy.mean(feat)
                            stdfeat = numpy.std(feat)
                        #dataset_exp_pre[expname]['mean_features'][feature][str(target)].append(meanfeat)
                        #dataset_exp_pre[expname]['std_features'][feature][str(target)].append(stdfeat)
                        dataset_exp_pre[expname]['mean_features'][feature][str(target)] = meanfeat
                        dataset_exp_pre[expname]['std_features'][feature][str(target)] = stdfeat


        # mean for all cells
        for i_exp, expname in enumerate(self.experiments):

            #collect everything in global structure
            self.dataset_processed[expname] = {}
            self.dataset_processed[expname]['amp'] = {}
            self.dataset_processed[expname]['amp_rel'] = {}
            self.dataset_processed[expname]['hypamp'] = []
            self.dataset_processed[expname]['ton'] = []
            self.dataset_processed[expname]['toff'] = []
            self.dataset_processed[expname]['features'] = {}

            for feature in self.features:
                self.dataset_processed[expname]['features'][feature] = {}
                for target in self.target_normap:
                    self.dataset_processed[expname]['features'][feature][str(target)] = []

            for target in self.target_normap:
                self.dataset_processed[expname]['amp'][str(target)] = []
                self.dataset_processed[expname]['amp_rel'][str(target)] = []

            for i_cell, cellname in enumerate(self.dataset):

                dataset_exp_pre = self.dataset[cellname]['experiments_preprocessed']

                hypamp = dataset_exp_pre[expname]['mean_hypamp']
                self.dataset_processed[expname]['hypamp'].append(hypamp)

                ton = dataset_exp_pre[expname]['mean_ton']
                self.dataset_processed[expname]['ton'].append(ton)

                toff = dataset_exp_pre[expname]['mean_toff']
                self.dataset_processed[expname]['toff'].append(toff)

                for target in self.target_normap:
                    amp = dataset_exp_pre[expname]['mean_amp'][str(target)]
                    self.dataset_processed[expname]['amp'][str(target)].append(amp)
                    amp_rel = dataset_exp_pre[expname]['mean_amp_rel'][str(target)]
                    self.dataset_processed[expname]['amp_rel'][str(target)].append(amp_rel)

                for feature in self.features:
                    for target in self.target_normap:
                        result = dataset_exp_pre[expname]['mean_features'][feature][str(target)]
                        self.dataset_processed[expname]['features'][feature][str(target)].append(result)

            #create means
            self.dataset_processed[expname]['mean_amp'] = {}
            self.dataset_processed[expname]['mean_amp_rel'] = {}
            self.dataset_processed[expname]['mean_features'] = {}
            self.dataset_processed[expname]['std_features'] = {}

            for feature in self.features:
                self.dataset_processed[expname]['mean_features'][feature] = {}
                self.dataset_processed[expname]['std_features'][feature] = {}

            hypamp = self.dataset_processed[expname]['hypamp']
            #self.dataset_processed[expname]['mean_hypamp'] = numpy.mean(numpy.concatenate((hypamp)))
            self.dataset_processed[expname]['mean_hypamp'] = numpy.mean(hypamp)


            ton = self.dataset_processed[expname]['ton']
            #self.dataset_processed[expname]['mean_ton'] = numpy.mean(numpy.concatenate((ton)))
            self.dataset_processed[expname]['mean_ton'] = numpy.mean(ton)

            toff = self.dataset_processed[expname]['toff']
            #self.dataset_processed[expname]['mean_toff'] = numpy.mean(numpy.concatenate((toff)))
            self.dataset_processed[expname]['mean_toff'] = numpy.mean(toff)

            for target in self.target_normap:
                amp = self.dataset_processed[expname]['amp'][str(target)]
                #self.dataset_processed[expname]['mean_amp'][str(target)] = numpy.mean(numpy.concatenate((amp)))
                self.dataset_processed[expname]['mean_amp'][str(target)] = numpy.mean(amp)

                amp_rel = self.dataset_processed[expname]['amp_rel'][str(target)]
                #self.dataset_processed[expname]['mean_amp_rel'][str(target)] = numpy.mean(numpy.concatenate((amp)))
                self.dataset_processed[expname]['mean_amp_rel'][str(target)] = numpy.mean(amp)

            for feature in self.features:
                for target in self.target_normap:
                    feat = self.dataset_processed[expname]['features'][feature][str(target)]

                    #self.dataset_processed[expname]['mean_features'][feature][str(target)] = numpy.mean(numpy.concatenate((feat)))
                    #self.dataset_processed[expname]['std_features'][feature][str(target)] = numpy.std(numpy.concatenate((feat)))

                    if self.use_nanmean:
                        self.dataset_processed[expname]['mean_features'][feature][str(target)] = numpy.nanmean(feat)
                        self.dataset_processed[expname]['std_features'][feature][str(target)] = numpy.nanstd(feat)
                    else:
                        self.dataset_processed[expname]['mean_features'][feature][str(target)] = numpy.mean(feat)
                        self.dataset_processed[expname]['std_features'][feature][str(target)] = numpy.std(feat)


    def plt_features(self):

        print "- plotting features"

        figs = collections.OrderedDict()

        tools.makedir(self.maindirname)

        markercyclercell = cycle(self.markerlist)
        colorcyclercell = cycle(self.colorlist)

        for i_cell, cellname in enumerate(self.dataset):

            dirname = self.maindirname+cellname
            tools.makedir(dirname)

            colorcell = next(colorcyclercell)
            markercell = next(markercyclercell)
            dataset_exp_pre = self.dataset[cellname]['experiments_preprocessed']

            for i_exp, expname in enumerate(dataset_exp_pre):

                if (i_cell == 0):
                    figname = "features_" + expname
                    plottools.tiled_figure(figname, frames=len(self.features), columns=3, figs=figs, dirname=self.maindirname,
                                            top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.3)


                figname = "features_" + cellname.split('/')[-1] + "_" + expname
                axs_cell = plottools.tiled_figure(figname, frames=len(self.features), columns=3, figs=figs, dirname=dirname,
                                        top=0.97, bottom=0.04, left=0.07, right=0.97, hspace=0.75, wspace=0.3)


                hypamp = numpy.mean(numpy.array(dataset_exp_pre[expname]['hypamp']))
                amp = dataset_exp_pre[expname]['amp']
                numspikes = dataset_exp_pre[expname]['features']['numspikes']
                feature_array = dataset_exp_pre[expname]['features']

                amp = list(itertools.chain.from_iterable(amp))
                amp_threshold, amp_rel = self.get_threshold(amp, numspikes)

                if self.relative:
                    amp_vals = numpy.array(amp)/amp_threshold * 100
                else:
                    amp_vals = numpy.array(amp)

                for fi, feature in enumerate(self.features):

                    dimensions = dataset_exp_pre[expname]['dimensions']
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

                        axs_cell[fi].plot(amp_vals0[is_not_nan], feat_vals0[is_not_nan], "", marker=marker, color=color, markersize=5, zorder=1, linewidth=1, markeredgecolor = 'none', clip_on=False)
                        axs_cell[fi].set_xticks(self.target_normap)
                        #axs_cell[fi].set_xticklabels(())
                        axs_cell[fi].set_title(feature)

                        figname = "features_" + expname
                        figs[figname]['axs'][fi].plot(amp_vals0[is_not_nan], feat_vals0[is_not_nan], "", marker=markercell, color=colorcell, markersize=5, zorder=1, linewidth=1, markeredgecolor = 'none', clip_on=False)
                        figs[figname]['axs'][fi].set_xticks(self.target_normap)
                        figs[figname]['axs'][fi].set_title(feature)

                    if 'mean_features' in dataset_exp_pre[expname]:

                        amp_rel_list = []
                        mean_list = []
                        std_list = []
                        for target in self.target_normap:
                            amp_rel_list.append(dataset_exp_pre[expname]['mean_amp_rel'][str(target)])
                            mean_list.append(dataset_exp_pre[expname]['mean_features'][feature][str(target)])
                            std_list.append(dataset_exp_pre[expname]['std_features'][feature][str(target)])

                        #mean_array = numpy.array(numpy.concatenate((mean_list)))
                        #std_array = numpy.array(numpy.concatenate((std_list)))
                        mean_array = numpy.array(mean_list)
                        std_array = numpy.array(std_list)
                        #amp_rel_array = numpy.array(numpy.concatenate((amp_rel_list)))
                        amp_rel_array = numpy.array(amp_rel_list)
                        #is_not_nan = ~numpy.isnan(mean_array)

                        e = axs_cell[fi].errorbar(amp_rel_array, mean_array, yerr=std_array, marker='s', color='k', linewidth=1, markersize=6, zorder=10, clip_on = False)
                        axs_cell[fi].set_xticks(self.target_normap)
                        for b in e[1]:
                            b.set_clip_on(False)


        for i_exp, expname in enumerate(self.experiments):

            if expname in self.dataset_processed:
                for fi, feature in enumerate(self.features):
                    amp_rel_list = []
                    mean_list = []
                    std_list = []
                    for target in self.target_normap:
                        amp_rel_list.append(self.dataset_processed[expname]['mean_amp_rel'][str(target)])
                        mean_list.append(self.dataset_processed[expname]['mean_features'][feature][str(target)])
                        std_list.append(self.dataset_processed[expname]['std_features'][feature][str(target)])

                    mean_array = numpy.array(mean_list)
                    std_array = numpy.array(std_list)
                    amp_rel_array = numpy.array(amp_rel_list)
                    #is_not_nan = ~numpy.isnan(mean_array)

                    figname = "features_" + expname
                    e = figs[figname]['axs'][fi].errorbar(amp_rel_list, mean_array, yerr=std_array, marker='s', color='k', linewidth=1, markersize=6, zorder=10, clip_on = False)
                    for b in e[1]:
                        b.set_clip_on(False)


        for i_fig, figname in enumerate(figs):
            fig = figs[figname]
            fig['fig'].savefig(fig['dirname'] + '/' + figname + '.pdf', dpi=300)
            plt.close(fig['fig'])


    def feature_config_all(self):
        self.create_feature_config(self.maindirname, self.dataset_processed)


    def feature_config_cells(self):
        for i_cell, cellname in enumerate(self.dataset):
            dataset_exp_pre = self.dataset[cellname]['experiments_preprocessed']
            self.create_feature_config(self.maindirname+cellname+'/', dataset_exp_pre)


    def create_feature_config(self, directory, dataset):

        print "- saving config files to " + directory

        self.stimulus_string = ""
        self.feature_string = ""

        for i_exp, expname in enumerate(self.experiments):
            if expname in dataset:
                for fi, feature in enumerate(self.features):
                    for it, target in enumerate(self.target_normap):

                        t = round(dataset[expname]['mean_amp_rel'][str(target)],2)
                        a = dataset[expname]['mean_amp'][str(target)]
                        h = dataset[expname]['mean_hypamp']
                        m = dataset[expname]['mean_features'][feature][str(target)]
                        s = dataset[expname]['std_features'][feature][str(target)]
                        ton = dataset[expname]['mean_ton']
                        toff = dataset[expname]['mean_toff']

                        if fi == 0:
                            self.stimulus_string += "configuration.addStimulus(opt.Stimulus(name=\"Step" + str(t) + "\", stimtype=\"SquarePulse\", amplitude=" + str(a) + ", hypamp=" + str(h) + ", delay=" + str(ton) + ", duration=" + str(toff-ton) + "))\n"

                        if ~numpy.isnan(m):
                            self.feature_string += "configuration.addFeature(opt.Feature(featuretype=\"" + feature + "\", stimulus=\"Step" + str(t) + "\", mean=" + str(m) + ", std=" + str(s) + ", weight=1))\n"

                self.feature_string += "\n"

                path = directory + "stimulus_output_" + expname + ".txt"
                with open(path, 'w') as text_file:
                    text_file.write(self.stimulus_string)
                    #print "- Saving ", path

                path = directory + "feature_output_" + expname + ".txt"
                with open(path, 'w') as text_file:
                    text_file.write(self.feature_string)
                    #print "- Saving ", path
