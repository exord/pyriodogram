from math import pi
import numpy as np

from . import linear_lstsq as fit
from .driftfuncs import offset, l, q

__all__ = ["periodogram",]

def periodogram(NU, DATA, FF0=[offset, l, q], covariates=None, NSIM=1):
    """
    Compute a periodogram for DATA on frequencies NU.
    The model includes a quadratic polynomial by default, and can include
    and indefinite series of additional covariates.

    :param array NU: frequencies [1/d] at which periodogram will be computed.

    :param list DATA: list containing data dictionaries; each element 
    of list corresponds to a given instrument, and a separate offset
    will be used in the model. Each dictionary must contain at least ('jdb',
    'vrad', 'svrad')

    :param list FF0: list of base function.

    :param list covariates: list of covariate arrays with covariats to 
    include in linear model. Each element of the list corresponds to a
    different instrument, and must be an array of dimensions (Ncov, Npoints),
    where Ncov is the number of covariates and Npoints is the number of 
    observations for this instrument, and must be equal to DATA[inst]['jdb']
    
    :param int NSIM: number of bootstrap permutations performed to estimate
    power significance.
    """
    
    if covariates is not None:
        ## Check if covariates are given in correct format.
    
        # Get number of covariates for each instrument.
        NCOVARIATES = [len(CC) for CC in covariates]
        assert NCOVARIATES.count(NCOVARIATES[0]) == len(NCOVARIATES), \
        ("Not all instruments have same number of covariates. This is not "
         "implemented yet. Sorry.")
    
    # Create global arrays with time, velocities and error
    x = np.concatenate([DATA[DD]['jdb'] for DD in DATA], axis=0)
    y = np.concatenate([DATA[DD]['vrad'] for DD in DATA], axis=0)
    ey = np.concatenate([DATA[DD]['svrad'] for DD in DATA], axis=0)

    if offset in FF0:
        # Create matrix with function values
        
        # For offset, use 0 and 1, depending on the instrument.
        FVALUES = [np.isin(x, DATA[DD]['jdb']).astype(int) for DD in DATA]

        # For the others, use x array
        FVALUES.extend([f(x) for f in FF0 if f != offset])

    else:
        # IF offset is not fit
        FVALUES = [f(x) for f in FF0 if f != offset]

        
    if covariates is not None:
        # Add a function for each covariate
        for i, DD in enumerate(DATA):  
            # Check if dimensions match.
            assert len(DATA[DD]['jdb']) == covariates[i].shape[1], \
            ('Covariates shape does not match number of points for '
             '{}'.format(DD))
  
        # Add a fuction for each covariate
        for i in range(covariates[0].shape[0]):
            FVALUES.append(np.concatenate([CC[i] for CC in covariates]))

    # Build array dictionary with concatenated arrays
    ALLDATA = {'jdb': x, 'vrad': y, 'svrad': ey}
    
    # Compute standard periodogram
    POW, S, PAR = periodogram_power(ALLDATA, NU, FVALUES)

    # If estimation of significance requested
    if NSIM > 1:
        PS = np.empty([NSIM, 2])

        NDATA = ALLDATA.copy()

        for i in range(NSIM):
            # Shuffle data indices
            ind = np.arange(len(ALLDATA['vrad']))
            np.random.shuffle(ind)

            NDATA['vrad'] = ALLDATA['vrad'][ind]
            NDATA['svrad'] = ALLDATA['svrad'][ind]

            # Shuffle covariates as well
            NFVALUES = list(FVALUES)
            for j in (-1 - np.arange(len(covariates))):
                NFVALUES[j] = FVALUES[j][ind]

            if (i+1)%10 == 0:
                print(i+1)
            POWi, Si = periodogram_power(NDATA, NU, NFVALUES)[:2]
            PS[i] = [np.max(POWi), np.max(-Si)]

        return POW, S, PAR, PS

    return POW, S, PAR, None


def periodogram_power(DATA, NU, FVALUES0):
    """
    Compute a periodogram for DATA on frequencies NU.
    The model includes a quadratic polynomial.
    """

    # Prepare arrays
    MODEL = np.empty([len(NU), len(DATA['jdb'])])
    CHI = np.empty(len(NU))

    # Fit null model (i.e., no sinusoid)
    P0 = fit.cuadminlin(DATA['vrad'], DATA['svrad'], FVALUES0)

    # Compute chi2 for null model
    MODEL0 = np.sum([P0[i] * FVALUES0[i] for i in range(len(P0))], axis=0)
    CHI0 = np.sum((DATA['vrad'] - MODEL0)**2 / DATA['svrad']**2)

    NPAR = len(FVALUES0) + 2
    PAR = np.empty([len(NU), NPAR])

    # For each frequency, fit sinusoid model
    for i, n in enumerate(NU):
        def s(t):
            return np.sin(2*pi*t*n)

        def c(t):
            return np.cos(2*pi*t*n)

        # Add sinusoid functions to function array
        FVALUES = list(FVALUES0)
        FVALUES.extend([s(DATA['jdb']), c(DATA['jdb'])])

        PAR[i] = fit.cuadminlin(DATA['vrad'], DATA['svrad'], FVALUES)

        # Compute chi2 for null model
        MODEL[i] = np.sum([PAR[i][j] * FVALUES[j]
                           for j in range(len(PAR[i]))], axis=0)

        CHI[i] = np.sum((DATA['vrad'] - MODEL[i])**2 / DATA['svrad']**2)

    POW = (CHI0 - CHI)/CHI0
    S = -0.5*CHI0 + 0.5*CHI - 0.5 * (len(FVALUES0) - len(FVALUES)
                                    ) * np.log(len(DATA['vrad']))
    S = -0.5*CHI0 + 0.5*CHI + np.log(len(DATA['vrad']))


    return POW, S, PAR
