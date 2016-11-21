# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:56:56 2016

@author: ttshimiz
"""

import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import astropy.modeling as apy_mod
from astropy.wcs import WCS
from spectral_cube import SpectralCube


def read_data(fn):
    """

    Reads in SINFONI FITS cube, cleans up the header, and returns a
    spectral-cube object.

    Parameters
    ----------
    fn = string; FITS file names

    Returns
    -------
    cube = spectral_cube.SpectralCube object

    """

    data = fits.getdata(fn)*1e-17
    header = fits.getheader(fn)

    # Check the spectral axis units and values
    # Switch to meters if the unit is micron
    cunit3 = header['CUNIT3']
    crval3 = header['CRVAL3']
    cdelt3 = header['CDELT3']

    if cunit3 == 'MICRON':
        cunit3 = 'meter'
        crval3 = crval3*10**-6
        cdelt3 = cdelt3*10**-6

    header['CUNIT3'] = cunit3
    header['CRVAL3'] = crval3
    header['CDELT3'] = cdelt3

    wcs = WCS(header)

    # Check now the cdelt3 value in the WCS object
    if wcs.wcs.cd[2, 2] != cdelt3:
        wcs.wcs.cd[2, 2] = cdelt3

    cube = SpectralCube(data=data, wcs=wcs, read_beam=False)

    # Convert to microns and add flux units
    cube = cube.with_spectral_unit(u.micron)
    cube._unit = u.W/u.meter**2/u.micron

    return cube
    

def cont_fit_single(x, spectrum, degree=1, errors=None, exclude=None):
    """
    Function to fit the continuum of a single spectrum with a polynomial.
    """
    
    if errors is None:
        errors = np.ones(len(spectrum))
        
    cont = apy_mod.models.Polynomial1D(degree=degree)
    
    # Use the endpoints of the spectrum to guess at zeroth and first order
    # parameters
    y1 = spectrum[0]
    y2 = spectrum[-1]
    x1 = x[0]
    x2 = x[-1]
    cont.c1 = (y2-y1)/(x2-x1)
    cont.c0 = y1 - cont.c1*x1
    
    fitter = apy_mod.fitting.LevMarLSQFitter()
    cont_fit = fitter(cont, x, spectrum, weights=1./errors)
    
    return cont_fit
    
    
    