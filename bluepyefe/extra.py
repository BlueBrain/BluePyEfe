"""Extra e-features functions"""

"""
Copyright (c) 2020, EPFL/Blue Brain Project

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

import numpy
import matplotlib.pyplot as plt


def spikerate_tau_jj(peaktimes):
    """Spike rate tau feature"""

    peaktimes = numpy.array(peaktimes)  # keep ms
    freq = 1. / numpy.diff(peaktimes)
    times = peaktimes[:-1]
    times = times - times[0]

    n = len(times)
    S = [0]
    for k in range(1, n):
        S.append(
            S[k - 1] + 1 / 2. * (freq[k] + freq[k - 1]) *
            (times[k] - times[k - 1]))
    S = numpy.array(S)

    M1 = sum((times - times[0])**2)
    M2 = sum((times - times[0]) * S)
    M3 = M2
    M4 = sum(S**2)

    V1 = sum((freq - freq[0]) * (times - times[0]))
    V2 = sum((freq - freq[0]) * S)

    invdet = 1 / (M1 * M4 - M3 * M2)
    I2 = invdet * -M2
    I4 = invdet * M1
    B1 = I2 * V1 + I4 * V2
    slope = B1
    tau = abs(-1. / slope)

    if tau > 1000.:
        tau = None

    return tau


def spikerate_drop(peaktimes, nstart=1, nend=2):
    """Spike rate drop feature"""

    peaktimes = numpy.array(peaktimes) * 1e-3  # convert from ms to s
    freq = 1. / numpy.diff(peaktimes)

    drop = numpy.mean(freq[-nend:]) / numpy.mean(freq[:nstart])

    return min(drop, 1.)


def spikerate_tau_log(peaktimes):
    """Spike rate tau log feature"""

    peaktimes = numpy.array(peaktimes)  # keep ms
    freq = 1. / numpy.diff(peaktimes)

    npeaks = len(peaktimes)
    used_isi = int(numpy.ceil(npeaks / 1.))

    # assume last frequency is f_inf (adapted frequency)
    # normalise by this value but do not include
    steady = freq[-1]

    freq = abs(freq - steady)[:used_isi + 1]
    times = peaktimes[0:used_isi + 1]
    log_freq = numpy.log(freq)

    i_use = []
    for i, f in enumerate(log_freq):
        if numpy.isfinite(f):
            i_use.append(i)
        else:
            break

    times = times[i_use]
    log_freq = log_freq[i_use]

    i_use = [0]
    for i, f in enumerate(numpy.diff(log_freq)):
        if f < 0:
            i_use.append(i + 1)
        else:
            break

    times = times[i_use]
    log_freq = log_freq[i_use]

    slope, _ = numpy.polyfit(times, log_freq, 1)

    tau = abs(-1. / slope)

    if tau > 1000.:
        tau = None

    return tau


def spikerate_slope(peaktimes):
    """Spike rate slope feature"""

    peaktimes = numpy.array(peaktimes) * 1e-3
    isi = numpy.diff(peaktimes)
    isi = isi
    x = numpy.arange(1, len(isi) + 1)

    times = peaktimes[:-1]
    log_isi = numpy.log(isi)
    log_x = numpy.log(x)
    log_times = numpy.log(times)
    freq = 1. / isi
    freq = freq - freq[-1]

    plt.subplot(5, 1, 1)
    plt.plot(times, isi, '*')
    plt.subplot(5, 1, 2)
    plt.plot(times, log_isi, '*')
    plt.subplot(5, 1, 3)
    plt.plot(log_times, log_isi, '*')
    plt.subplot(5, 1, 4)
    plt.plot(x, log_isi, '*')
    plt.subplot(5, 1, 5)
    plt.plot(log_x, log_isi, '*')
    plt.show()

    slope_log, _ = numpy.polyfit(log_x, log_isi, 1)
    slope_semilog, _ = numpy.polyfit(x, log_isi, 1)

    print(slope_log, slope_semilog)

    return slope_log


def spikerate_tau_slope(peaktimes):
    """Spike rate tau slope feature"""

    peaktimes = numpy.array(peaktimes) * 1e-3
    freq = 1. / numpy.diff(peaktimes)

    npeaks = len(peaktimes)
    used_isi = int(numpy.ceil(npeaks / 1.))

    # assume last frequency is f_inf (adapted frequency)
    # normalise by this value but do not include
    steady = freq[-1]

    freq = abs(freq - steady)[:used_isi + 1]
    times = peaktimes[0:used_isi + 1]
    log_freq = numpy.log(freq)

    i_use = []
    for i, f in enumerate(log_freq):
        if numpy.isfinite(f):
            i_use.append(i)
        else:
            break

    times = times[i_use]
    log_freq = log_freq[i_use]

    # i_use = [0]
    # for i, f in enumerate(numpy.diff(log_freq)):
    #     if f<0:
    #         i_use.append(i+1)
    #     else:
    #         break
    #
    # times = times[i_use]
    # log_freq = log_freq[i_use]

    plt.subplot(3, 1, 1)
    plt.plot(times, log_freq, '*')
    plt.subplot(3, 1, 2)
    plt.plot(times[1:], numpy.diff(log_freq), '*')
    plt.subplot(3, 1, 3)
    plt.plot(times[1:], numpy.diff(log_freq) / numpy.diff(times), '*')
    plt.show()

    tau = -numpy.mean(numpy.diff(log_freq) / numpy.diff(times))
    if numpy.isnan(tau):
        tau = 0

    return tau


def spikerate_tau_fit(peaktimes, nstart=1, nend=2):
    """Spike rate tau fit feature"""

    from scipy.optimize import curve_fit
    from functools import partial

    def func(x, tau, a, d):
        return a * numpy.exp(-x / tau) + d

    peaktimes = numpy.array(peaktimes)  # keep ms
    freq = 1. / numpy.diff(peaktimes)
    times = peaktimes[:-1]
    times = times - times[0]

    d = freq[-nend]
    a = freq[:nstart] - d

    func_ = partial(func, a=a, d=d)

    popt, pcov = curve_fit(func_, times, freq, p0=(1e9))

    tau = abs(popt[0])

    if tau > 1000.:
        tau = None

    return tau


def main():
    """Main"""

    tau = 200e-3
    fstart = 5.
    baseline = 10.
    tmax = 2.
    drop = baseline / (fstart + baseline)

    numpy.random.seed(1)
    peaktime = 1. / (fstart + baseline)
    peaktimes = [0]
    while peaktime < tmax:
        peaktimes.append(peaktime)
        next_freq = numpy.exp(-peaktime / tau) * fstart + baseline
        r = numpy.random.normal(0, 1e-9 * baseline, 1)[0]
        peaktime += 1. / (next_freq + r)
    peaktimes = numpy.array(peaktimes) + 200.

    print("target tau:", tau * 1e3)

    tau = spikerate_tau_jj(peaktimes * 1e3)  # convert from s to ms
    print("jj:", tau)

    tau = spikerate_tau_log(peaktimes * 1e3)  # convert from s to ms
    print("log:", tau)

    tau = spikerate_tau_fit(peaktimes * 1e3)  # convert from s to ms
    print("fit:", tau)

    tau = spikerate_slope(peaktimes * 1e3)  # convert from s to ms
    print("slope:", tau)

    print("target drop:", drop)
    drop = spikerate_drop(peaktimes * 1e3)  # convert from s to ms
    print(drop)

    f = 1. / numpy.diff(peaktimes)
    times = peaktimes[:-1]

    plt.figure()
    plt.plot(times, f, 'ko-')
    plt.show()


if __name__ == '__main__':
    main()
