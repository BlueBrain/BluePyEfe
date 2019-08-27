from scipy import stats
import matplotlib.pyplot as plt
import numpy


vals = numpy.arange(-2, 10., 0.01)

fig = plt.figure()
ax1 = fig.add_subplot(321)
x = stats.lognorm.rvs(0.8, size=100000) - 0.5


m_orig = numpy.mean(x)
std_orig = numpy.std(x)
scores_orig = abs(vals - m_orig) / std_orig

print('m_orig', m_orig, 'std_orig', std_orig)


prob = stats.probplot(x, dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')

ax1 = fig.add_subplot(322)
ax1.hist(
    x,
    1000,
    normed=True,
    histtype='stepfilled',
    color='b',
    edgecolor='none')
ax1.set_xlim((-1, 5))

ax2 = fig.add_subplot(323)

from rpy2.robjects.functions import SignatureTranslatedFunction
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

base = importr('base')
r_caret = importr("caret")
r_stats = importr("stats")
r_car = importr("car")

data = robjects.FloatVector(x)
fmla = robjects.Formula('x~1')
env = fmla.environment
env['x'] = data
fit = r_stats.lm(fmla)

lds = numpy.linspace(-2, 2, 21)
r_car.boxCox = SignatureTranslatedFunction(
    r_car.boxCox, {'r_lambda': 'lambda'})
yj = r_car.boxCox(fit,
                  family=robjects.StrVector(["yjPower"]),
                  r_lambda=robjects.FloatVector(lds),
                  plotit=False)

x_ = numpy.array(yj.rx('x')[0])
y_ = numpy.array(yj.rx('y')[0])


lmbda = x_[numpy.argmax(y_)]

print(x_)

from YeoJohnson import YeoJohnson

yj = YeoJohnson()
xt = yj.fit(x, lmbda)
valst = yj.fit(vals, lmbda)

m = numpy.mean(xt)
std = numpy.std(xt)
scores = abs(valst - m) / std


print('lmbda:', lmbda, 'mean:', m, 'std:', std)
print('mean in orig values:', yj.fit(numpy.array([m]), lmbda, inverse=True))

prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation')

ax3 = fig.add_subplot(324)
ax3.hist(
    xt,
    1000,
    normed=True,
    histtype='stepfilled',
    color='b',
    edgecolor='none')


ax4 = fig.add_subplot(325)
x = yj.fit(xt, lmbda, inverse=True)
ax4.plot(vals, scores_orig, 'r')
ax4.plot(vals, scores, 'b')

ax5 = fig.add_subplot(326)
ax5.hist(
    x,
    1000,
    normed=True,
    histtype='stepfilled',
    color='b',
    edgecolor='none')
ax5.set_xlim((-1, 5))


plt.show()
