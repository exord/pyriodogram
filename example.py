# -*- coding: utf-8 -*-

"""
Run a couple of tests on periodogram.
"""
import os
import numpy as np
import pylab as plt

from pyriodogram import periodogram, tools
from pyriodogram.driftfuncs import offset, l, q

homedir = os.getenv('HOME')

# Read data of multiple instruments.
data1 = tools.read_rdb('./data/file1.rdb')
data2 = tools.read_rdb('./data/file2.rdb')

# Maybe change rjd to jdb
for d in [data1, data2]:
    d['jdb'] = d.pop('rjd')

# DATA = [data1, data2]
DATA = {'d1': data1,
        'd2': data2}

TMIN = np.min([data2['jdb'].min(), data1['jdb'].min()])
TMAX = np.max([data2['jdb'].max(), data1['jdb'].max()])

NU = np.arange(1./(TMAX - TMIN), 1/1.7, 1/(TMAX - TMIN))

# Prepare covariates
covs = ['fwhm', 's_mw']

if len(covs) == 0:
    mycov = None
else:
    mycov = [np.array([DATA[inst][cov] for cov in covs]) for inst in DATA]


# Run only with offset for each instrument
POW, S, PAR, RR = periodogram(NU, DATA, FF0=[offset,], covariates=None)

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.semilogx(1/NU, POW, label='GLS', color='k')

# Run with offset and linear drift
POW, S, PAR, RR = periodogram(NU, DATA, FF0=[offset, l, q], covariates=None)
ax2 = fig.add_subplot(312)
ax2.semilogx(1/NU, POW, label='GLS + drift', color='C0')

# Run with offset, linear drift and covariates FWHM and S
POW, S, PAR, RR = periodogram(NU, DATA, FF0=[offset, l, q], covariates=mycov)
ax3 = fig.add_subplot(313)
ax3.semilogx(1/NU, POW, label='GLS + drift + FWHM + S', color='C2')

for ax in [ax1, ax2, ax3]:
    ax.legend(loc=0)
    ax.set_ylim(ax3.get_ylim())

plt.show()