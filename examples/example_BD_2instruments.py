# -*- coding: utf-8 -*-

"""
Run a couple of tests on periodogram.
"""
import os
import numpy as np

from pyriodogram import tools
from pyriodogram.driftfuncs import offset, l, q
from plot_BD import plot_BD

homedir = os.getenv('HOME')

def make_bd(target, drift, covariates, oversampling=1.0, save=True):
    
    datadir = os.path.join(homedir, 'ExP', target, 'data')
    
    # Check if both datafiles exist
    datafile1 = os.path.join(datadir, target+'_HARPS-DRS-3-5.rdb')
    datafile2 = os.path.join(datadir, target+'_HARPS-DRS-3-5b.rdb')
    
    for df in (datafile1, datafile2):
        assert os.path.exists(df), '{} does not exist.'.format(df)
    
    # Read data of multiple instruments.
    data1 = tools.read_rdb(os.path.join(datafile1))
    data2 = tools.read_rdb(os.path.join(datafile2))
    
    # Maybe change rjd to jdb
    for d in [data1, data2]:
        try:
            d['jdb'] = d.pop('rjd')
        except KeyError:
            if 'jdb' in d:
                continue
            else:
                print('rjd / jdb key not found!')
    
    # DATA = [data1, data2]
    DATA = {'d1': data1,
            'd2': data2}
    
    # Crete frequency array based on sampling.
    TMIN = np.min([data2['jdb'].min(), data1['jdb'].min()])
    TMAX = np.max([data2['jdb'].max(), data1['jdb'].max()])
    
    NU = np.arange(1./(TMAX - TMIN), 1/1.7, 1/(TMAX - TMIN)/oversampling)
        
    # Prepare covariates
    if len(covariates) == 0:
        mycov = None
    else:
        mycov = [np.array([DATA[inst][cov] for cov in covariates]
                         ) for inst in DATA]
    
    fig, maxper = plot_BD(NU, DATA, mycov, covariates, drift=drift)
    
    # Add target name to figure
    fig.texts[0].set_text(target+'; '+fig.texts[0].get_text())
    
    if save:
        # Save plot
        plotdir = os.path.join(homedir, 'ExP', target, 'plots')
        if not os.path.exists(plotdir):
            os.mkdir(plotdir)
        fig.savefig(os.path.join(plotdir, target+'_BD_oversamp{}'
                                 '.pdf'.format(int(oversampling))))
    
    # Redo periodogram, using an additional sinusoid at the period of the main peak
    signal1 = []
    for inst in DATA:
        signal1.append(np.array([np.cos(DATA[inst]['jdb']*2*np.pi/maxper), 
                                 np.sin(DATA[inst]['jdb']*2*np.pi/maxper)]))
        
    fig2, maxper2 = plot_BD(NU, DATA, mycov, covariates, signals=signal1,
                            drift=drift)
    
    # Add target name to figure
    fig2.texts[0].set_text(fig.texts[0].get_text()+'; '+fig2.texts[0].get_text())
    
    if save:
        fig2.savefig(os.path.join(plotdir, target+'_k1_BD_oversamp{}'
                                  '.pdf'.format(int(oversampling))))
    
    return


if __name__ == '__main__':
    # Drift functions
    drift = [offset, l, q]

    for target in ['HD38858', 'HD126525', 'HD150433', 'HD157172', 'HD215456']:
        print(target)
        make_bd(target, drift=drift, covariates=['s_mw', 'fwhm', 'bis_span'],
                oversampling=8.0, save=False)
        