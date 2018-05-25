# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt

from pyriodogram import periodogram
from pyriodogram.driftfuncs import offset, l, q

def plot_BD(NU, DATA, covariates, covnames, signals=None, 
            drift=[offset, l, q]):
    # Prepare figure
    fig = plt.figure(figsize=(10,5))
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.5)

    ax1 = plt.subplot2grid((3, 5), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 5), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 5), (2, 0), colspan=2)

    axs = [ax1, ax2, ax3]
        
    # Run only with offset for each instrument
    POW, S, PAR, RR = periodogram(NU, DATA, FF0=[offset,], 
                                  covariates=signals)
    maxpow = POW.max()
    maxax = ax1

    ax1.semilogx(1/NU, POW, label='GLS', color='0.5')

    # Run with offset and linear drift
    POW, S, PAR, RR = periodogram(NU, DATA, FF0=drift, covariates=signals)
    ax2.semilogx(1/NU, POW, label='GLS + drift', color='0.25')
    if maxpow < POW.max():
        maxax = POW.max()
        maxax = ax2
        
    # Add signals as covariates
    if signals is not None:
        for inst in range(len(covariates)):
            covariates[inst] = np.concatenate([covariates[inst], 
                                              signals[inst]])
        covnames.extend(['cos1', 'sin1'])
    
    # Run with offset, linear drift and covariates
    POW, S, PAR, RR = periodogram(NU, DATA, FF0=drift, covariates=covariates)
    label = 'GLS + drift + '+'{} + '*(len(covnames)-1)+'{}'
    ax3.semilogx(1/NU, POW, label=label.format(*covnames), color='0.1')
    ax3.set_xlabel('Period [d]')
    if maxpow < POW.max():
        maxax = POW.max()
        maxax = ax3

    # Find highestpeak in full model
    maxper = 1/NU[np.argmax(POW)]

    # Make plot of covariates
    for i in range(len(covnames)):
        print(i, covnames[i])

        # Plot periodogram and histogram of covariate power
        ax = plt.subplot2grid((len(covnames), 5), (i, 2), colspan=2)
        ax.semilogx(1/NU, PAR[:,len(drift)+i], color='C{}'.format(i+2), 
                              label=covnames[i].upper())
    
        axh = plt.subplot2grid((len(covnames), 5), (i, 4))
        axh.hist(PAR[:,len(drift)+i], 100, color='C{}'.format(i+2), 
                     label=covnames[i])
    
        # Remove ticks in histogram
        axh.yaxis.set_major_formatter(plt.NullFormatter())
    
        ax.legend(loc=0)
        ax.axvline(maxper, ls=':', color='r')
    
        if i == len(covnames) - 1:
            ax.set_xlabel('Period [d]')
    
        # Annotate maximum period
        # maxax.annotate(s='{:.2f} d'.format(maxper), xy=(maxper, maxpow))

    # Decorate plots.
    # Maximum period, legend, and align vertical axis of periodograms.
    for ax in axs:
        ax.axvline(maxper, ls=':', color='r')
        ax.legend(loc=0, fontsize=8)
        ax.set_ylim(maxax.get_ylim())
        if ax != axs[-1]:
            ax.xaxis.set_major_formatter(plt.NullFormatter())

    # Figure title
    title = '{:.2f} d'.format(maxper)
    fig.text(0.5, 0.95, title, va='center', ha='center', fontsize=16)
           
    return fig, maxper

