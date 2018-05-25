# -*- coding: utf-8 -*-

"""
Run a couple of tests on periodogram.
"""
import os
import numpy as np

from pyriodogram import tools
from pyriodogram.driftfuncs import offset, l, q, c
from plot_BD import plot_BD

homedir = os.getenv('HOME')

def make_bd(target, drift, covariates, oversampling=1.0, save=True):
    
    datadir = os.path.join(homedir, 'ExP', target, 'data')
    
    # Check if both datafiles exist
    datafile = os.path.join(datadir, target+'_NAIRA_corr_th_2_cti.rdb')
    
    assert os.path.exists(datafile), '{} does not exist.'.format(datafile)
    
    # Read data of multiple instruments.
    print(datafile)
    data1 = tools.read_rdb(datafile)
    
    # Maybe change rjd to jdb
    try:
        data1['jdb'] = data1.pop('rjd')
    except KeyError:
        if 'jdb' in data1:
            pass
        else:
            print('rjd / jdb key not found!')
    
    # DATA = [data1, data2]
    DATA = {'d1': data1,
            }
    
    # Crete frequency array based on sampling.
    TMIN = np.min(data1['jdb'])
    TMAX = np.max(data1['jdb'])
    
    NU = np.arange(1./(TMAX - TMIN), 1/1.7, 1/(TMAX - TMIN)/oversampling)
        
    # Prepare covariates
    if covariates is None:
        mycov = covariates
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
    drift = [offset, l, q, c]

    # for target in ['Gl251', 'Gl378', 'Gl411', 'Gl338A']:
    for target in ['Gl251',]:
        print(target)
        make_bd(target, drift=drift, covariates=['fwhm', 'bis_span'],
                oversampling=8.0, save=False)
        