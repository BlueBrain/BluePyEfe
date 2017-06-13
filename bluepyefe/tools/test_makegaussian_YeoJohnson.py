from scipy import stats
import matplotlib.pyplot as plt
import numpy


vals = numpy.arange(-2, 10., 0.01)

fig = plt.figure()
ax1 = fig.add_subplot(321)
x = stats.lognorm.rvs(0.8, size=100000)-0.5
#x = stats.loggamma.rvs(6, size=10000)
#x = stats.norm.rvs(0.8, size=100000)+10
#x = stats.beta.rvs(2,8, size=100000)
#x = stats.beta.rvs(1,9, size=100000)


m_orig = numpy.mean(x)
std_orig = numpy.std(x)
scores_orig = abs(vals - m_orig)/std_orig

print 'm_orig', m_orig, 'std_orig', std_orig


prob = stats.probplot(x, dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')

ax1 = fig.add_subplot(322)
ax1.hist(x, 1000, normed=True, histtype='stepfilled', color='b', edgecolor = 'none')
ax1.set_xlim((-1,5))

ax2 = fig.add_subplot(323)

from rpy2.robjects.functions import SignatureTranslatedFunction
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

base = importr('base')
r_caret = importr("caret")
r_stats = importr("stats")
r_car = importr("car")


# r_caret.preProcess = SignatureTranslatedFunction(r_caret.preProcess, {'r_lambda': 'lambda'})
# r_scale = r_caret.preProcess(robjects.DataFrame({'a': robjects.FloatVector(x)}),
#                           method=robjects.StrVector(["YeoJohnson"]),
#                           r_lambda=robjects.FloatVector(numpy.arange(-1,1,0.1)),
#                           na_remove=False)
#
# a = r_scale.rx('yj')[0].rx('a')[0].rx('lambda')[0].rx('Y1')[0]
#
# temp = r_stats.predict(r_scale, robjects.DataFrame({'a': robjects.FloatVector(feature_in)}))
# xt = numpy.array(temp.rx2(1), dtype=numpy.float64)
#
# temp = r_stats.predict(r_scale, robjects.DataFrame({'a': robjects.FloatVector(vals)}))
# valst = numpy.array(temp.rx2(1), dtype=numpy.float64)
# print type(a), a

import rpy2.robjects as robjects

data = robjects.FloatVector(x)
fmla = robjects.Formula('x~1')
env = fmla.environment
env['x'] = data
fit = r_stats.lm(fmla)

r_car.boxCox = SignatureTranslatedFunction(r_car.boxCox, {'r_lambda': 'lambda'})
yj = r_car.boxCox(fit,
            family = robjects.StrVector(["yjPower"]),
            r_lambda = robjects.FloatVector(numpy.arange(-2,2,0.1)),
            plotit = False)

x_ = numpy.array(yj.rx('x')[0])
y_ = numpy.array(yj.rx('y')[0])

lmbda = x_[numpy.argmax(y_)]


from YeoJohnson import YeoJohnson

yj = YeoJohnson()
xt = yj.fit(x, lmbda)
valst = yj.fit(vals, lmbda)

m = numpy.mean(xt)
std = numpy.std(xt)
scores = abs(valst - m)/std


print 'lmbda:', lmbda, 'mean:', m, 'std:', std
print 'mean in orig values:', yj.fit(numpy.array([m]), lmbda, inverse=True)

prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation')

ax3 = fig.add_subplot(324)
ax3.hist(xt, 1000, normed=True, histtype='stepfilled', color='b', edgecolor = 'none')


ax4 = fig.add_subplot(325)
x = yj.fit(xt, lmbda, inverse=True)
ax4.plot(vals, scores_orig, 'r')
ax4.plot(vals, scores, 'b')

ax5 = fig.add_subplot(326)
ax5.hist(x, 1000, normed=True, histtype='stepfilled', color='b', edgecolor = 'none')
ax5.set_xlim((-1,5))


plt.show()
