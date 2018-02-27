from math import pi
import numpy as np

from . import linear_lstsq as fit
from .driftfuncs import gamma, l, q

__all__ = ["periodogram",]

def periodogram(NU, DATA, FF0=[gamma, l, q], covariates=None, NSIM=1):
    """
    Compute a periodogram for DATA on frequencies NU.
    The model includes a quadratic polynomial by default, and can include
    and indefinite series of additional covariates.

    :param array NU: frequencies [1/d] at which periodogram will be computed.
    :param dict DATA: data dictionary; must contain at least ('jdb', 'vrad',
    'svrad')
    :param list FF0: list of base function.
    :param list covariates: list of covariates to include in linear model.
    Each one must be an array with same length as DATA['jdb'].
    :param int NSIM: number of bootstrap permutations performed to estimate
    power significance.
    """

    FVALUES = [f(DATA['jdb']) for f in FF0]

    # Add a function for each covariate
    for x in covariates:
        FVALUES.append(x)

    # Compute standard periodogram
    POW, S, PAR = periodogram_power(DATA, NU, FVALUES)

    # If estimation of significance requested
    if NSIM > 1:
        PS = np.empty([NSIM, 2])

        NDATA = DATA.copy()

        for i in range(NSIM):
            # Shuffle data indices
            ind = np.arange(len(DATA['vrad']))
            np.random.shuffle(ind)

            NDATA['vrad'] = DATA['vrad'][ind]
            NDATA['svrad'] = DATA['svrad'][ind]

            # Shuffle covariates as well
            NFVALUES = list(FVALUES)
            for j in (-1 - np.arange(len(covariates))):
                NFVALUES[j] = FVALUES[j][ind]

            print(i)
            POWi, Si = periodogram_power(NDATA, NU, NFVALUES)[:2]
            PS[i] = [np.max(POWi), np.max(-Si)]
            #PS[i] = [POWi, Si]

        return POW, S, PS

    return POW, S, None


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
