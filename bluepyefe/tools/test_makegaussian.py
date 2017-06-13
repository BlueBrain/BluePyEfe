from scipy import stats
import matplotlib.pyplot as plt
import numpy

def invboxcox(y, ld, shift=None):
    if ld == 0:
        val = numpy.exp(y)
    else:
        val = numpy.exp(numpy.log(ld*y+1)/ld)

    if shift is not None:
        val -= shift

    return val


def make_nonzero_positive(y):
    # make values non zero and positive

    minimum = min(y)
    if minimum <= 0:
        shift = max(y) + abs(minimum) + 1e-9
    else:
        shift = 0.

    y = y + shift

    return y, shift


def clip_ld_convert(vals, shift, ld):
    # convert values with lambda

    vals = vals + shift # add shift to values
    vals = vals.clip(min=1e-9) # clip just before zero if shift was not enough
    vals_bc = stats.boxcox(vals, ld)

    return vals_bc


vals = numpy.arange(0, 10., 0.01)

fig = plt.figure()
ax1 = fig.add_subplot(321)
#x = stats.lognorm.rvs(0.8, size=100000)
x = stats.loggamma.rvs(6, size=10000)
#x = stats.norm.rvs(0.8, size=100000)+10
#x = stats.beta.rvs(2,8, size=100000)
#x = stats.beta.rvs(1,9, size=100000)


m_orig = numpy.mean(x)
std_orig = numpy.std(x)
scores_orig = abs(vals - m_orig)/std_orig

print numpy.median(x) #, numpy.mode(x)

print 'm_orig', m_orig, 'std_orig', std_orig

x, shift = make_nonzero_positive(x)

print "shift", shift

prob = stats.probplot(x, dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')

ax1 = fig.add_subplot(322)
ax1.hist(x, 1000, normed=True, histtype='stepfilled', color='b', edgecolor = 'none')


ax2 = fig.add_subplot(323)
xt, ld = stats.boxcox(x)

# convert values with lambda
vals_bc = clip_ld_convert(vals, shift, ld)

m = numpy.mean(xt)
std = numpy.std(xt)
scores = abs(vals_bc - m)/std

print 'ld:', ld, 'mean:', m, 'std:', std
print 'mean in orig values:', invboxcox(numpy.array([m]), ld, shift)

prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation')

ax3 = fig.add_subplot(324)
ax3.hist(xt, 100, normed=True, histtype='stepfilled', color='b', edgecolor = 'none')


ax4 = fig.add_subplot(325)
#x = invboxcox(xt,ld)
#prob = stats.probplot(x, dist=stats.norm, plot=ax4)

ax4.plot(vals, scores_orig, 'r')
ax4.plot(vals, scores, 'b')


#ax5 = fig.add_subplot(326)
#ax5.hist(x, 100, normed=True, histtype='stepfilled', color='b', edgecolor = 'none')


plt.show()
