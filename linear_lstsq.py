#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module containing basic functionalities for linear least squares.

Created on Tue Feb 27 19:07:30 2018

@author: rodrigo
"""
import numpy as np

def cuadminlin(y, ey, f):
    """
    Hace cuadrados minProduce las matrices necesarias para correr
    scipy.linalg.lstsq.
    La funcion del modelo es a*f, donde f es el array de las funciones de base
    evaluadas en t x
    """

    At = np.array(f)
    A = At.T

    # Original, slower
    #V = np.diag(1./ey**2.)
    #MA = np.dot(At, np.dot(V, A))
    #MB = np.dot(At, np.dot(V, y.reshape(len(y), 1)))

    MA = np.dot(At, A / ey[:, None]**2)
    MB = np.dot(At, (y/ey**2)[:, None])

    # Solve equation to get parameters
    par = np.linalg.solve(MA, MB)

    # Invert array MA to compute parameter covariance
    #MAi = np.linalg.inv(MA)

    #return np.array(par)[:,0]#, np.sqrt(np.diag(MAi)), MAi
    return np.array(par)[:, 0]

def cuadminlin_fullcov(y, V, f):
    """
    Least squares with full covariance for data.

    :param array V: data covariance matrix.
    """
    At = np.array(f)
    A = At.T

    x = np.linalg.solve(V, y)
    z = np.linalg.solve(V, A)
    MA = np.dot(At, z)
    MB = np.dot(At, x)

    # Solve equation to get parameters
    par = np.linalg.solve(MA, MB)

    # Invert array MA to compute parameter covariance
    #MAi = np.linalg.inv(MA)

    #return np.array(par)[:,0]#, np.sqrt(np.diag(MAi)), MAi
    return par
