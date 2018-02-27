# -*- coding: utf-8 -*-

# Define functions for polyonmial RV drift model
def l(t):
    # Linear drift, in kms/year
    return (t - t.mean())/365.25

def q(t):
    # Quadratic drift in kms**2/year
    return (t - t.mean())**2 / 365.25**2

def c(t):
    # Cubic drift in kms**3/year
    return (t - t.mean())**3 / 365.25**3

def gamma(t):
    return t*0.0 + 1.

