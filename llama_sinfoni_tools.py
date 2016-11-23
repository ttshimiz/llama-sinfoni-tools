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

    cube = SpectralCube(data=data, wcs=wcs, read_beam=False, meta={'BUNIT':'W / (m2 micron)'})

    # Convert to microns
    cube = cube.with_spectral_unit(u.micron)

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
    
    
def remove_cont(cube, degree=1, exclude=None):
    """
    Function to loop through all of the spectra in a cube and subtract out the continuum
    """
    
    xsize = cube.shape[1]
    ysize = cube.shape[2]
    nparams = degree+1
    fit_params = np.zeros((xsize, ysize, nparams))
    spec_ax = cube.spectral_axis.value
    data_cont_remove = np.zeros(cube.shape)
    
    for i in range(xsize):
        for j in range(ysize):
            
            spec = cube[:, i, j].value/10**(-17)
            
            if np.any(~np.isnan(spec)):
                cont = cont_fit_single(spec_ax, spec, degree=degree, exclude=exclude)
            
                for n in range(nparams):
                    fit_params[i, j, n] = cont.parameters[n]
                
                data_cont_remove[:, i, j] = (spec - cont(spec_ax))*10**(-17)

            else:
				fit_params[i, j, :] = np.nan
				data_cont_remove[:, i, j] = np.nan
    
    cube_cont_remove = SpectralCube(data=data_cont_remove, wcs=cube.wcs,
                                    meta={'BUNIT':cube.unit.to_string()})
    cube_cont_remove = cube_cont_remove.with_spectral_unit(cube.spectral_axis.unit)
    
    return cube_cont_remove, fit_params
    
    
def gauss_fit_single(x, spectrum, guess=None, errors=None, exclude=None):
    """
    Function to fit a single spectrum with a Gaussian
    """
    
    if errors is None:
         errors = np.ones(len(spectrum))
         
    model = apy_mod.models.Gaussian1D()
    model.amplitude.min = 0
    model.stddev.min = 0
     
    if guess is None:
        model.amplitude = np.max(spectrum)
        model.mean = x[np.argmax(spectrum)]
        model.stddev = 3*(x[1] - x[0])
    else:
        model.amplitude = guess[0]
        model.mean = guess[1]
        model.stddev = guess[2]
    
    
    fitter = apy_mod.fitting.LevMarLSQFitter()     
    gauss_fit = fitter(model, x, spectrum, weights=1./errors)
    
    return gauss_fit
        
def cubefit_gauss(cube, guess=None, exclude=None):
    """
    Function to loop through all of the spectra in a cube and fit a gaussian
    """
    
    xsize = cube.shape[1]
    ysize = cube.shape[2]
    flux_unit = cube.unit
    spec_ax = cube.spectral_axis.value
    spec_ax_unit = cube.spectral_axis.unit
    
    fit_params = {'amplitude': np.zeros((xsize, ysize))*flux_unit,
                  'mean': np.zeros((xsize, ysize))*spec_ax_unit,
                  'sigma': np.zeros((xsize, ysize))*spec_ax_unit}
    
    for i in range(xsize):
        for j in range(ysize):
            
            spec = cube[:, i, j].value/10**(-17)
            
            if np.any(~np.isnan(spec)):
                gauss_mod = gauss_fit_single(spec_ax, spec, guess=guess, exclude=exclude)
            
                fit_params['amplitude'][i, j] = gauss_mod.amplitude.value*flux_unit
                fit_params['mean'][i, j] = gauss_mod.mean.value*spec_ax_unit
                fit_params['sigma'][i, j] = gauss_mod.stddev.value*spec_ax_unit
                
            else:
				fit_params['amplitude'][i, j] = np.nan
				fit_params['mean'][i, j] = np.nan
				fit_params['sigma'][i, j] = np.nan
        
